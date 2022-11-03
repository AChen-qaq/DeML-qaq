NUM_GPUS=8

python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} single_task_fine_tune.py \
    --dataset_name super_glue \
    --dataset_config_name rte \
    --template_name "can we infer" \
    --model_name_or_path /home/yanan/shaonan/pretrained_model/t5-large-lm-adapt \
    --output_dir /home/yanan/shaonan/t-zero/exp_finetune_official_rte_can_we_infer \
    --input_eos \
    --max_length 384 \
    --target_max_length 32 \
    --per_device_train_batch_size 2 \
    --num_train_epochs 4
