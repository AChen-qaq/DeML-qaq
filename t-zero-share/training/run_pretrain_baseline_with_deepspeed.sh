NUM_GPUS=8

deepspeed --num_gpus=${NUM_GPUS} pretrain_t0_with_trainer.py \
  --do_train \
  --do_eval \
  --deepspeed /home/yanan/shaonan/t-zero/config/deepspeed/ds_config_zero2_bf16.json \
  --train_task_list /home/yanan/shaonan/t-zero/config/pretraining/train_cla.list \
  --data_dir /home/yanan/shaonan/data/preprocessed_baseline \
  --train_file /home/yanan/shaonan/data/combined_train_src_384_tgt_32_maxsize_250000.json \
  --eval_file /home/yanan/shaonan/data/combined_eval_src_384_tgt_32_maxsize_250000.json \
  --output_dir /home/yanan/shaonan/t-zero/exp_pretrain_baseline_bf16 \
  --template_dir /home/yanan/shaonan/t-zero/templates_feature \
  --model_name_or_path /home/yanan/shaonan/pretrained_model/t5-large-lm-adapt \
  --max_seq_length 384 \
  --max_tgt_length 32 \
  --evaluation_strategy steps \
  --eval_steps 2000 \
  --per_device_train_batch_size 12 \
  --per_device_eval_batch_size 48 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 2000 \
  --save_total_limit 3 \
  --report_to tensorboard






