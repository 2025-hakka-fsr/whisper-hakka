python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch-char/checkpoint-2600" \
--temp_ckpt_folder "temp-char" \
--language zh \
--eval_datasets output_data-char/test \
--device 0 \
--batch_size 8 \
--output_dir predictions_dir-char
