

export CUDA_VISIBLE_DEVICES=0
python run_all_eval.py \
    --test_split /home/yanan/shaonan/t-zero/config/pretraining/temp.list \
    --model_name_or_path /home/yanan/shaonan/t-zero/exp_finetune_v1 \
    --template_dir /home/yanan/shaonan/t-zero/templates_feature \
    --output_dir ./result_finetune_v1
