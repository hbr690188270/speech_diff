python generate.py \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --num_train_epochs 10 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 3000 \
    --lr_scheduler_type linear \
    --output_dir /mnt/data/bairu/repos/story_gen/output/eval \
    --data_dir /mnt/data/bairu/repos/speech_diff/datasets/ \
    --model_cache_dir /mnt/data/bairu/model_cache/ \
    --seed 42 \
    --preprocessing_num_workers 5 \
    --checkpointing_steps 500 \
    --gpu_ids 0 \
    --num_gpu 1 \
    --sentence_level \
    --train_decoder \
    --report_to wandb \
    --run_name libri_kl5 \
    --encoder_path output/libri_kl5/epoch=49-step=318500.ckpt
