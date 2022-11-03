NUM_GPUS=8

python -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} pretrain_t0_with_trainer.py \
  --do_train \
  --do_eval \
  --train_task_list /home/yanan/shaonan/t-zero/config/pretraining/train_cla.list \
  --data_dir /home/yanan/shaonan/data/preprocessed_baseline \
  --train_file /home/yanan/shaonan/data/combined_train_src_384_tgt_32_maxsize_250000.json \
  --eval_file /home/yanan/shaonan/data/combined_eval_src_384_tgt_32_maxsize_250000.json \
  --output_dir /home/yanan/shaonan/t-zero/exp_pretrain_basleine \
  --template_dir /home/yanan/shaonan/t-zero/templates_feature \
  --model_name_or_path /home/yanan/shaonan/pretrained_model/t5-large-lm-adapt \
  --max_seq_length 384 \
  --max_tgt_length 32 \
  --evaluation_strategy steps \
  --eval_steps 10000 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --logging_steps 10 \
  --save_steps 20000 \
  --report_to tensorboard






