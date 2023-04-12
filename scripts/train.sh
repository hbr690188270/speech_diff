python train_ae.py \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-5 \
    --weight_decay 0.001 \
    --num_train_epochs 50 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 3000 \
    --lr_scheduler_type linear \
    --output_dir /mnt/data/bairu/repos/story_gen/output/libri_kl5 \
    --data_dir /mnt/data/bairu/repos/speech_diff/datasets/ \
    --model_cache_dir /mnt/data/bairu/model_cache/ \
    --seed 42 \
    --preprocessing_num_workers 5 \
    --checkpointing_steps 500 \
    --gpu_ids 0 1 \
    --num_gpu 2 \
    --sentence_level \
    --train_decoder \
    --report_to wandb \
    --run_name libri_kl5 \
    # --use_kl \
    # --kl_weight 1e-5 \

