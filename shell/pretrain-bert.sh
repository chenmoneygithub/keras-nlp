python3 examples/bert/bert_train.py \
    --input_files bert-pretraining-data \
    --read_from_gcs True \
    --gcs_bucket=chenmoney-testing-east \
    --vocab_file bert_vocab_uncased.txt \
    --model_size base \
    --num_train_steps=500000 \
    --checkpoint_save_directory="bert-training/saved_checkpoints/bert_base_500K" \
    --skip_restore=False \
    --saved_model_output="bert-training/saved_models/bert_base_500K" \