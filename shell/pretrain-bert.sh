python3 examples/bert/bert_train.py \
    --input_files bert-pretraining-data-512-76/testing \
    --read_from_gcs True \
    --gcs_bucket=chenmoney-testing \
    --vocab_file bert_vocab_uncased.txt \
    --model_size tiny \
    --num_train_steps=100 \
    --checkpoint_save_directory="saved_checkpoints/bert_test" \
    --skip_restore=True \
    --saved_model_output="saved_models/bert_test" \