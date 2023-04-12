import numpy as np
import torch
import torch.nn as nn
from transformers import DefaultDataCollator

class MyDataCollator(DefaultDataCollator):
    def __init__(self, encoder_tokenizer, decoder_tokenizer):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
    
    def __call__(self, features):
        
        batch = {}

        tmp = [torch.tensor(f['input_ids']) for f in features]
        batch['input_ids'] = nn.utils.rnn.pad_sequence(tmp, batch_first=True, padding_value=self.encoder_tokenizer.pad_token_id)
        
        tmp = [torch.tensor(f['attention_mask']) for f in features]
        batch['attention_mask'] = nn.utils.rnn.pad_sequence(tmp, batch_first=True, padding_value=0)


        length_list = [len(x['decoder_input_ids']) for x in features]
        max_length = max(length_list)
        all_decoder_input_ids = []
        all_decoder_labels = []
        for idx in range(len(features)):
            curr_length = len(features[idx]['decoder_input_ids'])
            pad_input_ids = [self.decoder_tokenizer.eos_token_id] + features[idx]['decoder_input_ids'] + [self.decoder_tokenizer.pad_token_id] * (max_length - curr_length)
            pad_labels = features[idx]['decoder_input_ids'] + [self.decoder_tokenizer.eos_token_id] + [self.decoder_tokenizer.pad_token_id] * (max_length - curr_length)
            all_decoder_input_ids.append(pad_input_ids)
            all_decoder_labels.append(pad_labels)
        
        decoder_input_ids = np.array(all_decoder_input_ids)
        decoder_input_ids = torch.from_numpy(decoder_input_ids).long()
        
        decoder_labels = np.array(all_decoder_labels)
        decoder_labels = torch.from_numpy(decoder_labels).long()
        
        batch['decoder_input_ids'] = decoder_input_ids
        batch['decoder_labels'] = decoder_labels

        return batch

