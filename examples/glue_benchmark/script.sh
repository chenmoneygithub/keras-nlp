for task in qnli qqp; do
  python3 glue.py --task_name="$task" --submission_directory="glue_submissions/" --epochs=3 --learning_rate=5e-5
done

python3 glue.py --task_name="mnli_matched" --batch_size=32  --submission_directory="glue_submissions/" --epochs=2 --learning_rate=5e-5 --save_finetuning_model="saved/mnli"

python3 glue.py --task_name="ax" --batch_size=32  --submission_directory="glue_submissions/" --epochs=3 --learning_rate=5e-5 --load_finetuning_model="saved/mnli"
python3 glue.py --task_name="mnli_mismatched" --batch_size=32  --submission_directory="glue_submissions/" --epochs=3 --learning_rate=5e-5 --load_finetuning_model="saved/mnli"
