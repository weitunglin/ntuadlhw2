from datasets import load_dataset
from icecream import ic
import numpy as np

dataset = load_dataset('ntuadlhw2-data', data_files={'train': 'train.jsonl', 'validation': 'public.jsonl'})

dataset['train'] = dataset['train'].add_column("maintext_length", [len(i) for i in dataset['train']['maintext']])

ic(dataset['train'])

length = np.array(dataset['train']['maintext_length'])
ic(np.mean(length))
ic(np.median(length))
ic(np.std(length))
ic(np.max(length))
ic(np.min(length))