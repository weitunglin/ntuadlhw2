import sys
from itertools import chain
from copy import deepcopy

import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, SummarizationPipeline
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

class CustomPipeline(SummarizationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _parse_and_tokenize(self, *args, truncation, max_length):
        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        if isinstance(args[0], list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )
        inputs = self.tokenizer(*args, padding=padding, max_length=max_length, truncation=truncation, return_tensors=self.framework)
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs

pipe_kwargs = {
    'truncation': True,
    'max_length': 64,
    'num_beams': 30,
    'no_repeat_ngram_size': 2,
    'early_stopping': True
}

debug = False
batch_size = 8
model_path = 'weitung8/ntuadlhw2'
dataset_path = '.'
max_source_length = 256
padding = False

input_file = sys.argv[1]
output_file = sys.argv[2]

model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

datasets = load_dataset(dataset_path, data_files={'test':input_file})
ids = deepcopy(datasets['test']['id'])
remove_columns = list(filter(lambda x: x != 'maintext', datasets['test'].column_names))
datasets = datasets.map(lambda x: x, remove_columns=remove_columns, batched=True, num_proc=32, desc='Running tokenizer on dataset')

if debug:
    print(len(datasets['test']['maintext']))

def postprocess_text(preds):
    preds = [{'summary_text': pred['summary_text'].strip()} for pred in preds]

    # rougeLSum expects newline after each sentence
    preds = [{'summary_text': "\n".join(nltk.sent_tokenize(pred['summary_text']))} for pred in preds]

    return preds

dataloader = DataLoader(datasets['test'], batch_size=batch_size)
#pipe = pipeline('summarization', model=model, tokenizer=tokenizer, device='cuda')
pipe = CustomPipeline(model=model, tokenizer=tokenizer, device='cuda', batch_size=batch_size)
pipe._preprocess_params = {**pipe._preprocess_params, 'max_length':max_source_length}
pipe.model.config.prefix = ''
if debug:
    print(pipe.model.config.prefix)
    print(pipe._preprocess_params)
all_summaries = []

if debug:
    def show_memory(text=''):
        t = torch.cuda.get_device_properties(0).total_memory
        c = torch.cuda.memory_cached(0)
        a = torch.cuda.memory_allocated(0)
        f = c-a  # free inside cache
        print(f'\n\n{text}\nTotal: {t/1e9:.2f}\nCached: {c/1e9:.2f} \nAllocated: {a/1e9:.2f} \nFree in cache: {f/1e9:.2f}\n\n')

with torch.no_grad():
    for batch in tqdm(dataloader):
        #show_memory()
        summaries = pipe(batch['maintext'], **pipe_kwargs)
        summaries = postprocess_text(summaries)
        all_summaries.extend(summaries)
        #torch.cuda.empty_cache()

if debug:
    print(all_summaries[:10])
    print(len(ids))
    print(len(all_summaries))

result = pd.DataFrame(data={'id':ids, 'title':[i['summary_text'] for i in all_summaries]})

result.to_json(output_file, orient='records', lines=True)

