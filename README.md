# ntuadlhw2

## experiments

### train model

```python train.py --model_name_or_path google/mt5-small --dataset_name ntuadlhw2-data --text_column maintext --learning_rate 2e-5 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --num_train_epochs 5 --gradient_accumulation_steps 32 --max_source_length 256 --max_target_length 64 --val_max_target_length 64 --use_adafactor --seed 41 --output_dir weitung8/ntuadlhw2 --push_to_hub --hub_model_id weitung8/ntuadlhw2 --with_tracking --report_to wandb --num_beams 30 --checkpointing_steps epoch --num_train_epochs_to_eval 1 --num_warmup_epochs_to_eval 3 --lr_scheduler_type cosine --weight_decay 3e-4```

### train model using multi gpu

```TOKENIZERS_PARALLESIM=false accelerate launch train.py --model_name_or_path google/mt5-small --dataset_name ntuadlhw2-data --text_column maintext --learning_rate 2e-5 --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --num_train_epochs 5 --gradient_accumulation_steps 32 --max_source_length 256 --max_target_length 64 --val_max_target_length 64 --use_adafactor --seed 41 --output_dir weitung8/ntuadlhw2 --push_to_hub --hub_model_id weitung8/ntuadlhw2 --with_tracking --report_to wandb --num_beams 30 --checkpointing_steps epoch --num_train_epochs_to_eval 1 --num_warmup_epochs_to_eval 3 --lr_scheduler_type cosine --weight_decay 3e-4```

### eval using different generators

```python eval.py --model_name_or_path weitung8/ntuadlhw2 --dataset_name ntuadlhw2-data --text_column maintext --per_device_eval_batch_size 2 --max_source_length 256 --max_target_length 64 --val_max_target_length 64 --seed 41 --output_dir weitung8/ntuadlhw2```
