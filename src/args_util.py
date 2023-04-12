import argparse
from transformers import SchedulerType

def parse_args():
    parser = argparse.ArgumentParser(description="Multinomial diffusion model for text generation.")
    # model parameters
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--data_dir", type=str, default=None, help="path to ROC dataset.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help=(
            'wandb name'
        ),
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0,)
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        help="Path to save pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument("--num_gpu", type=int, default=1,)
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=[0])    
    
    parser.add_argument("--sentence_level", action = 'store_true')
    parser.add_argument("--batch_model", action = 'store_true')
    parser.add_argument("--cpg", action = 'store_true')
    parser.add_argument("--use_kl", action = 'store_true')
    parser.add_argument("--encoder_path", type = str,)
    parser.add_argument("--train_decoder", action = 'store_true')
    parser.add_argument("--kl_weight", type=float, default=1e-6,)
    parser.add_argument("--kl_scheduler", type = str, default = 'linear', choices = ['linear', 'cyclic'])

    args = parser.parse_args()
    return args
