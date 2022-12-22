for task in cola mrpc rte stsb qnli qqp sst2; do
  python3 glue.py --task_name="$task" --submission_directory="glue_submissions/" --batch_size=32  --tpu_name="local" --epochs=3 --learning_rate=2e-5 --submission_directory="glue_submissions/"
done

python3 glue.py --task_name="mnli_matched" \
    --submission_directory="glue_submissions/" \
    --save_finetuning_model="saved/mnli" --batch_size=32  --tpu_name="local" --epochs=3 --learning_rate=2e-5 --submission_directory="glue_submissions/"

python3 glue.py --task_name="mnli_mismatched" \
    --submission_directory="glue_submissions/" \
    --load_finetuning_model="saved/mnli" --batch_size=32  --tpu_name="local" --epochs=3 --learning_rate=2e-5 --submission_directory="glue_submissions/"

python3 glue.py --task_name="ax" \
    --submission_directory="glue_submissions/" \
    --load_finetuning_model="saved/mnli" --batch_size=32  --tpu_name="local" --epochs=3 --learning_rate=2e-5 --submission_directory="glue_submissions/"