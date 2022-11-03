

export CUDA_VISIBLE_DEVICES=1,2
python run_all_eval.py \
    --test_split /mfs/shaonan/moonshot/t-zero/config/setting_5/test_temp.list \
    --model_name_or_path /mfs/shaonan/pretrained_model/T0_3B \
    --parallelize \
    --dataset_type ga \
    --template_dir /mfs/shaonan/moonshot/t-zero/evaluation/ga_t0_norm_shot/ga_configs/step_1 \
    --output_dir ./debug \
    --debug

#    --template_dir /mfs/shaonan/moonshot/data/temp_dir \
#    --template_dir /mfs/shaonan/moonshot/t-zero/evaluation/ga_t0_norm_shot/ga_configs/step_1 \
