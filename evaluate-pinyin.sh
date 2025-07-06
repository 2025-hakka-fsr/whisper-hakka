python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch-pinyin/checkpoint-2600" \
--temp_ckpt_folder "temp-pinyin" \
--language en \
--eval_datasets formatted_data_pinyin/eval_dataset_sm \
--device 0 \
--batch_size 8 \
--output_dir predictions_dir-pinyin
