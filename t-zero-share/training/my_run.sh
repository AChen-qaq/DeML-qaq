NUM_GPUS=1
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --master_port 1234 --nproc_per_node ${NUM_GPUS} finetune_t0.py \
  --train_task_list /localdata/DeML/t-zero-share/config/pretraining/single.list \
  --data_dir /localdata/DeML/data \
  --output_dir /localdata/DeML/t-zero-share/superglue_boolq_baseline \
  --template_dir /localdata/DeML/t-zero-share/templates_feature \
  --model_name_or_path /share/huggingface_models/t5-large-lm-adapt \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-3 \
  --num_train_epochs 5 \
  --eval_interval_steps 1000 \
  --start_eval_steps 0 \
  --max_length 384 \
  --target_max_length 32 \
  --num_warmup_steps 0