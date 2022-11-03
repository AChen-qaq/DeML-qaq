NUM_GPUS=1

python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} pretrain_t0.py \
  --train_task_list /home/yanan/shaonan/t-zero/config/pretraining/train_cla.list \
  --data_dir /home/yanan/shaonan/data/preprocessed_baseline \
  --train_file /home/yanan/shaonan/data/combined_train_src_384_tgt_32_maxsize_250000.json \
  --eval_file /home/yanan/shaonan/data/combined_eval_src_384_tgt_32_maxsize_250000.json \
  --output_dir /home/yanan/shaonan/t-zero/exp_pretrain_baseline \
  --template_dir /home/yanan/shaonan/t-zero/templates_feature \
  --model_name_or_path /home/yanan/shaonan/pretrained_model/t5-large-lm-adapt \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 12 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --eval_interval_steps 1000 \
  --start_eval_steps 0 \
  --max_length 384 \
  --target_max_length 32 \
  --num_warmup_steps 0