import sys
from itertools import chain
from copy import deepcopy

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

pipe_kwargs = {
    'max_length': 64,
    'num_beams': 30,
    'no_repeat_ngram_size': 2,
    'early_stopping': True
}

debug = True
batch_size = 2
model_path = 'weitung8/ntuadlhw2'
dataset_path = '.'
max_source_length = 256
padding = False

input_file = sys.argv[1]
output_file = sys.argv[2]

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.cuda()
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

def preprocess_function(examples):
    inputs = examples['maintext']
    inputs = ['summaries: ' + inp for inp in inputs]
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    return model_inputs

datasets = load_dataset(dataset_path, data_files={'test':input_file})
ids = deepcopy(datasets['test']['id'])
remove_columns = list(filter(lambda x: x != 'maintext', datasets['test'].column_names))
datasets = datasets.map(preprocess_function, remove_columns=remove_columns, batched=True, num_proc=16, desc='Running tokenizer on dataset')

if debug:
    print(len(datasets['test']['maintext']))

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

dataloader = DataLoader(datasets['test'], collate_fn=data_collator, batch_size=batch_size)

all_summaries = []

if debug:
    def show_memory(text=''):
        t = torch.cuda.get_device_properties(0).total_memory
        c = torch.cuda.memory_cached(0)
        a = torch.cuda.memory_allocated(0)
        f = c-a  # free inside cache
        print(f'\n\n{text}\nTotal: {t/1e9:.2f}\nCached: {c/1e9:.2f} \nAllocated: {a/1e9:.2f} \nFree in cache: {f/1e9:.2f}\n\n')

for batch in tqdm(dataloader):
    with torch.no_grad():
        show_memory()
        genreated_tokens = model.generate(batch['input_ids'], **pipe_kwargs)
        summaries = tokenizer.decode_batch(genreated_tokens, skip_special_tokens=True)
        print(summaries[0])
        all_summaries.extend(summaries)
        torch.cuda.empty_cache()

if debug:
    print(all_summaries[:10])

result = pd.DataFrame(data={'id':ids, 'title':[i['summary_text'] for i in summaries]})

result.to_json(output_file, orient='records', lines=True)

