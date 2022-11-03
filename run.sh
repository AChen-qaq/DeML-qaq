export TASK_NAME=cola

python run_glue_no_trainer.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-3 \
  --num_train_epochs 30 \
  --output_dir /tmp/${TASK_NAME}_fullSGD \
  --with_tracking