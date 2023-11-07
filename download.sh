mkdir ntuadlhw2-data

curl -L -o 'ntuadlhw2-data/train.jsonl' 'https://huggingface.co/datasets/weitung8/ntuadlhw2/resolve/main/train.jsonl'
curl -L -o 'ntuadlhw2-data/public.jsonl' 'https://huggingface.co/datasets/weitung8/ntuadlhw2/resolve/main/valid.jsonl'
curl -L -o 'ntuadlhw2-data/sample_test.jsonl' 'https://huggingface.co/datasets/weitung8/ntuadlhw2/resolve/main/sample_test.jsonl'

mkdir weitung8/ntuadlhw2

curl -L -o 'weitung8/ntuadlhw2/config.json' 'https://huggingface.co/weitung8/ntuadlhw2/resolve/main/config.json'
curl -L -o 'weitung8/ntuadlhw2/generation_config.json' 'https://huggingface.co/weitung8/ntuadlhw2/resolve/main/generation_config.json'
curl -L -o 'weitung8/ntuadlhw2/pytorch_model.bin' 'https://huggingface.co/weitung8/ntuadlhw2/resolve/main/pytorch_model.bin'
curl -L -o 'weitung8/ntuadlhw2/special_tokens_map.json' 'https://huggingface.co/weitung8/ntuadlhw2/resolve/main/special_tokens_map.json'
curl -L -o 'weitung8/ntuadlhw2/spiece.model' 'https://huggingface.co/weitung8/ntuadlhw2/resolve/main/spiece.model'
curl -L -o 'weitung8/ntuadlhw2/tokenizer.json' 'https://huggingface.co/weitung8/ntuadlhw2/resolve/main/tokenizer.json'
curl -L -o 'weitung8/ntuadlhw2/tokenizer_config.json' 'https://huggingface.co/weitung8/ntuadlhw2/resolve/main/tokenizer_config.json'

