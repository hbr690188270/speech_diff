import numpy as np
import os
from typing import List, Optional, Tuple, Union, Dict, Callable

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.modules.activation import MultiheadAttention
import torch.nn.functional as F
import lightning as L

from transformers import RobertaModel, GenerationConfig, GPT2LMHeadModel
from transformers.optimization import get_scheduler
from transformers import NoBadWordsLogitsProcessor, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from .metric import token_level_accuracy
from .distribution import DiagonalGaussianDistribution
from .utils import frange_cycle_linear

class MyGPT2ForCausalLM(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
    
    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, past_key_values=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update (
            {
            # "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        )
        return model_inputs

def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class CPGLayer(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(self, x: torch.Tensor, mem: torch.Tensor,
                   attn_mask: Optional[torch.Tensor], key_padding_mask: Optional[torch.Tensor], is_causal: bool = False) -> torch.Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class CpgGPT2ForCausalLM(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        config = self.transformer.config
        dim_feedforward = config.n_inner if config.n_inner is not None else 4 * config.hidden_size
        # dropout = config.embd_pdrop
        dropout = 0
        self.cpg_layer = CPGLayer(d_model = config.hidden_size, nhead = config.num_attention_heads, dim_feedforward = dim_feedforward, 
                                  dropout = dropout, activation = "gelu", layer_norm_eps = config.layer_norm_epsilon,
                                  batch_first = True, norm_first = True)
        self._init_weights(self.cpg_layer)

    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, outline_embedding = None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "outline_embedding": outline_embedding,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        outline_embedding: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        
        if outline_embedding is not None:
            hidden_states = self.cpg_layer(tgt = hidden_states, memory = outline_embedding, )

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )


class SentenceLevelAE(L.LightningModule):
    def __init__(self, encoder_name = "voidism/diffcse-roberta-base-sts", encoder_cache_dir = None,
                       decoder_name = "EleutherAI/gpt-neo-2.7B", decoder_cache_dir = None, 
                       freeze_decoder = True,
                       learning_rate = 1e-5,
                       weight_decay = 1e-4,
                       lr_scheduler_type = 'cosine',
                       warmup_steps = 500,
                       num_training_steps = None,
                       include_decoder = True,
                       use_kl = False,
                       kl_weight = 1e-5,
                       kl_scheduler = 'linear',
                       cpg = False,
                       pretrained_path = None,
                       ) -> None:
        super().__init__()
        print(f"AutoEncoder: sentence level")
        self.encoder = RobertaModel.from_pretrained(encoder_name, cache_dir = encoder_cache_dir, output_loading_info = False)
        self.use_kl = use_kl
        self.max_kl_weight = kl_weight
        self.kl_step = 0
        self.min_kl_step = 5000
        self.max_kl_step = 20000  ## after 10000 steps KL weight increase to max weight
        self.kl_scheduler = kl_scheduler
        self.cpg = cpg
        assert self.kl_scheduler in ['linear', 'cyclic']
        if self.kl_scheduler == 'cyclic':
            print("using cyclic for kl weight!")
            self.kl_schedule_array = frange_cycle_linear(start = 0, stop = 1, n_epoch = num_training_steps, n_cycle = 20, ratio = 0.75)
        if "gpt-neo" in decoder_name:
            self.decoder_hidden_size = 2560
        else:
            self.decoder_hidden_size = 768
        if self.use_kl:
            self.mu_head = nn.Linear(self.encoder.config.hidden_size, self.decoder_hidden_size)
            self.var_head = nn.Linear(self.encoder.config.hidden_size, self.decoder_hidden_size)
        else:
            self.encoder_proj = nn.Linear(self.encoder.config.hidden_size,
                self.decoder_hidden_size)
        self.freeze_decoder = freeze_decoder

        if include_decoder:
            if self.cpg:
                print("using cpg to add outline embeddings...")
                self.decoder = CpgGPT2ForCausalLM.from_pretrained(decoder_name, cache_dir = decoder_cache_dir)
            else:
                self.decoder = MyGPT2ForCausalLM.from_pretrained(decoder_name, cache_dir = decoder_cache_dir)
            self.decoder.config.pad_token_id = self.decoder.config.vocab_size
            self.decoder.resize_token_embeddings(self.decoder.config.vocab_size + 1)

            if self.use_kl:
                self.decoder._init_weights(self.mu_head)
                self.decoder._init_weights(self.var_head)
            
            else:
                self.decoder._init_weights(self.encoder_proj)
            if self.freeze_decoder:
                if self.cpg:
                    self.decoder.transformer.requires_grad_(False)
                    self.decoder.lm_head.requires_grad_(False)
                else:
                    self.decoder.requires_grad_(False)
            self.logit_processor = NoBadWordsLogitsProcessor(bad_words_ids = [[self.decoder.config.pad_token_id]], eos_token_id = self.decoder.config.eos_token_id)
        else:
            self.decoder = None
            self.logit_processor = None

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location = self.encoder.device)['state_dict']
            self.load_state_dict(state_dict, strict = False)
            del state_dict
            torch.cuda.empty_cache()

        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps


    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        scheduler = get_scheduler(name = self.lr_scheduler_type, optimizer = optimizer,
                                num_warmup_steps = self.warmup_steps, num_training_steps = self.num_training_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_save_checkpoint(self, checkpoint):
        # 99% of use cases you don't need to implement this method
        if self.freeze_decoder:
            state_dict_keys = checkpoint['state_dict'].keys()
            keys_to_rm = [x for x in state_dict_keys if (x.startswith('decoder') and not x.startswith("decoder.cpg_layer"))]
            for key in keys_to_rm:
                checkpoint['state_dict'].pop(key)

    def encode(self, inputs):
        sent_emb = self.encoder(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask']).last_hidden_state
        raw_sent_emb = sent_emb[:, 0, :]
        
        if self.use_kl:
            mu = self.mu_head(raw_sent_emb) # (B, 768)
            logvar = self.var_head(raw_sent_emb) # (B, 768)
            posterior = DiagonalGaussianDistribution(mu, logvar, deterministic=False)
            return posterior, raw_sent_emb
        else:
            sent_emb = self.encoder_proj(raw_sent_emb)
            return sent_emb, raw_sent_emb
    
    def forward(self, inputs):
        return None

    def shared_forward(self, inputs, batch_idx, splits = 'train'):
        if self.use_kl:
            posterior, raw_sent_emb = self.encode(inputs)
            sentence_embeddings = posterior.sample()
        else:
            sentence_embeddings, raw_sent_emb = self.encode(inputs)  # batch_size, sentence num, hidden


        batch_size, hidden_dim = sentence_embeddings.size()
        
        if self.cpg:
            decoder_inputs_ids = inputs['decoder_input_ids'] ## start with <eos>
            decoder_labels = inputs['decoder_labels'] ## end with <eos>
            sentence_length = decoder_inputs_ids.size(1)
            outputs = self.decoder(input_ids = decoder_inputs_ids, outline_embedding = sentence_embeddings)

            logits = outputs.logits
            selected_logits = logits
            labels = decoder_labels
        else:
            decoder_inputs_ids = inputs['decoder_input_ids'][:, 1:]  # batch_size, sentence_length - 1,   1: to remove <eos>
            decoder_labels = inputs['decoder_labels'] ## end with <eos>
            decoder_input_embeds = self.decoder.transformer.wte(decoder_inputs_ids)  # batch_size, sentence_length - 1, hidden

            sentence_embeddings = sentence_embeddings.view(batch_size , 1, hidden_dim)
            prefix_embeds = torch.cat([sentence_embeddings, decoder_input_embeds], dim = 1)
            outputs = self.decoder(inputs_embeds=prefix_embeds)

            logits = outputs.logits

            selected_logits = logits.contiguous()
            labels = decoder_labels.contiguous()  # batch_size, sentence_length

        loss_fct = CrossEntropyLoss(ignore_index = self.decoder.config.pad_token_id)
        loss = loss_fct(selected_logits.view(-1, self.decoder.config.vocab_size), labels.view(-1))
        _,_,acc = token_level_accuracy(labels, self.decoder.config.pad_token_id, selected_logits)
        sync_dist = splits in ['valid', 'test']

        if self.use_kl:
            kl_loss = posterior.kl().mean()
            if self.kl_scheduler == 'cyclic':
                kl_ratio = self.kl_schedule_array[self.kl_step]
                kl_weight = kl_ratio * self.max_kl_weight
            else:
                if self.kl_step < self.min_kl_step:
                    kl_weight = 0
                elif self.kl_step < self.max_kl_step:
                    kl_weight = self.kl_step / self.max_kl_step * self.max_kl_weight
                else:
                    kl_weight = self.max_kl_weight
            total_loss = loss + kl_loss * kl_weight
            self.log(f"{splits}_loss", loss.item(), prog_bar = True, sync_dist = sync_dist)
            self.log(f"{splits}_acc", acc, prog_bar = True, sync_dist = sync_dist)
            self.log(f"lr", self.lr_schedulers().get_last_lr()[0], prog_bar = True, sync_dist = sync_dist)
            if splits == 'train':
                self.log(f"klweight", kl_weight, prog_bar = True, sync_dist = sync_dist)
            self.log(f"{splits}_kl", kl_loss.item(), prog_bar = True, sync_dist = sync_dist)
    
            return total_loss

        else:
            self.log(f"{splits}_loss", loss.item(), prog_bar = True, sync_dist = sync_dist)
            self.log(f"{splits}_acc", acc, prog_bar = True, sync_dist = sync_dist)
            self.log(f"lr", self.lr_schedulers().get_last_lr()[0], prog_bar = True, sync_dist = sync_dist)

            return loss

    def training_step(self, inputs, batch_idx):
        if self.use_kl:
            if self.trainer.is_global_zero:
                self.kl_step += 1
        return self.shared_forward(inputs, batch_idx, 'train')

    def test_step(self, inputs, batch_idx):
        return self.shared_forward(inputs, batch_idx, 'test')

    def validation_step(self, inputs, batch_idx):
        return self.shared_forward(inputs, batch_idx, 'valid')

    def generate(self, sentence_embeddings = None, max_length = 50):    
        batch_size, hidden_dim = sentence_embeddings.size()
        
        generation_config = GenerationConfig(
            max_new_tokens=max_length, do_sample=False, eos_token_id=50256, bos_token_id=50256, pad_token_id = 50257, num_beams=4, early_stopping=True, use_cache = True,
        )
        if self.cpg:
            outputs = self.decoder.generate(outline_embedding = sentence_embeddings,
                        generation_config = generation_config, logits_processor = LogitsProcessorList([self.logit_processor]))
        else:
            outputs = self.decoder.generate(inputs_embeds=sentence_embeddings.reshape(batch_size, 1, hidden_dim),
                        generation_config = generation_config, logits_processor = LogitsProcessorList([self.logit_processor]))

        return outputs

