NUM_GPUS=8

for gradient_accumulation_steps in 1 2 3 4 5
do
  echo now using ${gradient_accumulation_steps} gradient_accumulation_steps
deepspeed --num_gpus=${NUM_GPUS} pretrain_t0_with_trainer.py \
  --do_train \
  --do_eval \
  --deepspeed /home/yanan/shaonan/t-zero/config/deepspeed/ds_config_zero2_bf16.json \
  --train_task_list /home/yanan/shaonan/t-zero/config/pretraining/temp.list \
  --data_dir /home/yanan/shaonan/data/preprocessed_baseline \
  --train_file /home/yanan/shaonan/data/testsubset_train_src_384_tgt_32_maxsize_250000.json \
  --eval_file /home/yanan/shaonan/data/testsubset_eval_src_384_tgt_32_maxsize_250000.json \
  --output_dir /home/yanan/shaonan/t-zero/exp_finetune_3B_${gradient_accumulation_steps} \
  --template_dir /home/yanan/shaonan/t-zero/templates_feature \
  --model_name_or_path /home/yanan/shaonan/pretrained_model/t5-xl-lm-adapt \
  --max_seq_length 384 \
  --max_tgt_length 32 \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --per_device_train_batch_size 5 \
  --per_device_eval_batch_size 12 \
  --learning_rate 5e-5 \
  --gradient_accumulation_steps ${gradient_accumulation_steps} \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 500 \
  --load_best_model_at_end \
  --save_total_limit 3 \
  --group_by_length \
  --report_to tensorboard
done