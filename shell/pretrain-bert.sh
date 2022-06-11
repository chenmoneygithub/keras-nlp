python3 examples/bert/bert_train.py \
    --input_files bert-pretraining-data \
    --read_from_gcs True \
    --gcs_bucket=chenmoney-testing-east \
    --vocab_file bert_vocab_uncased.txt \
    --model_size base \
    --num_train_steps=500000 \
    --checkpoint_save_directory="saved_checkpoints/bert_base_500K" \
    --skip_restore=True \
    --saved_model_output="saved_models/bert_base_500K" \