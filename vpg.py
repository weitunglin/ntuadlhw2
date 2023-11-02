import sys
import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
import evaluate
import nltk
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch import tensor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from icecream import ic
import torch.nn.functional as F

import tensorflow as tf
devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(devices[0], True)
    tf.config.experimental.per_process_gpu_memory_fraction(0.5)
    print("Success in setting memory growth")
except:
    print("Failed to set memory growth, invalid device or cannot modify virtual devices once initialized.")
from tw_rouge import get_rouge

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
)
from transformers.utils import is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version


class PolicyEstimator():
    def __init__(self, env):
        self.num_observations = env.observation_space.shape[0]
        self.num_actions = env.action_space.n

        self.network = nn.Sequential(
            nn.Linear(self.num_observations, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_actions),
            nn.Softmax(dim=-1)
        )

    def predict(self, observation):
        return self.network(torch.FloatTensor(observation))

def vanilla_policy_gradient(num_episodes=1500, batch_size=10, discount_factor=0.99, render=False,
                            early_exit_reward_amount=None):

    vocab_size = 250112
    total_rewards, batch_rewards, batch_observations, batch_actions = [], [], [], []
    batch_counter = 1
    debug = True

    raw_datasets = load_dataset('ntuadlhw2-data', data_files={'train':'train.jsonl', 'valid':'public.jsonl'})

    if debug == True:
        raw_datasets['train'] = raw_datasets['train'].select(range(100))
        raw_datasets['valid'] = raw_datasets['valid'].select(range(20))

    config = AutoConfig.from_pretrained('weitung8/ntuadlhw2', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('weitung8/ntuadlhw2', use_fast=True, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained('weitung8/ntuadlhw2', config=config)

    text_column = 'maintext'
    summary_column = 'title'
    prefix = ''

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=256, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=64, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    column_names = raw_datasets['train'].column_names
    train_dataset = raw_datasets["train"].map(
        preprocess_function,
        batched=True,
        num_proc=32,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    # Temporarily set max_target_length for validation.
    max_target_length = 64
    eval_dataset = raw_datasets["valid"].map(
        preprocess_function,
        batched=True,
        num_proc=32,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    if torch.cuda.is_available():
        model.to('cuda')

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=1
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=1)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    action_space = np.arange(vocab_size)

    for current_episode in range(num_episodes):
        #observation = env.reset()
        model.train()
        rewards, actions, observations = [], [], []

        for step, batch in tqdm(enumerate(train_dataloader)):
            batch.to('cuda')
            outputs = model(**batch, output_hidden_states=True)

            if debug:
                ic(batch.keys())
                for (k, v) in outputs.items():
                    ic(k)
                    if isinstance(v, tuple):
                        for x in v:
                            if isinstance(x, torch.Tensor):
                                print(x.size())
                            else:
                                ic(type(x))
                    if isinstance(v, torch.Tensor):
                        ic(v.size())

            with torch.no_grad():
                def get_reward(preds, labels):
                    rewards = []
                    for pred, label in zip(preds, labels):
                        rouge = get_rouge(pred, label)
                        reward = (((rouge['rouge-1']['f'] + rouge['rouge-2']['f']) / 2) + rouge['rouge-l']['f']) / 2
                        rewards.append(reward)
                    return rewards
                gen_kwargs = {
                    "max_length": 64,
                    "num_beams": 15,
                    "no_repeat_ngram_size": 2,
                    "early_stopping": True
                }
                generated_tokens = model.generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                """
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=0
                )
                # If we did not pad to max length, we need to pad the labels too
                labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=0)
                """
                labels = batch["labels"]

                generated_tokens = generated_tokens.cpu().numpy()
                labels = labels.cpu().numpy()

                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, 0)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                if debug:
                    ic(decoded_preds)
                    ic(decoded_labels)
                rewards = get_reward(decoded_preds, decoded_labels)
                ic(rewards)

            batch_observations.extend(outputs['encoder_hidden_states'])
            batch_actions.extend(outputs['decoder_hidden_states'])
            batch_rewards.extend([0.0 for i in range(len(outputs['decoder_hidden_states'])-1)])
            batch_rewards.extend(rewards)

            if debug:
                ic(batch_rewards[-9])
                ic(len(batch_observations))
                ic(len(batch_actions))
                ic(len(batch_rewards))

            batch_counter += 1
            total_rewards.append(sum(rewards))

            if batch_counter >= batch_size:
                # reset gradient
                optimizer.zero_grad()

                # tensorify things
                batch_rewards = torch.FloatTensor(torch.tensor(batch_rewards))
                batch_observationss = torch.FloatTensor(torch.tensor(batch_observations))
                batch_actions = torch.LongTensor(torch.tensor(batch_actions))

                # calculate loss
                logprob = torch.log(estimator.predict(batch_observations))
                batch_actions = batch_actions.reshape(len(batch_actions), 1)
                selected_logprobs = batch_rewards * torch.gather(logprob, 1, batch_actions).squeeze()
                loss = -selected_logprobs.mean()

                # backprop/optimize
                loss.backward()
                optimizer.step()

                # reset the batch
                batch_rewards, batch_observations, batch_actions = [], [], []
                batch_counter = 1

            # get running average of last 100 rewards, print every 100 episodes
            average_reward = np.mean(total_rewards[-100:])
            if current_episode % 100 == 0:
                print(f"average of last 100 rewards as of episode {current_episode}: {average_reward:.2f}")

            # quit early if average_reward is high enough
            if early_exit_reward_amount and average_reward > early_exit_reward_amount:
                return total_rewards

    return total_rewards

if __name__ == '__main__':
    # actually run the algorithm
    rewards = vanilla_policy_gradient(num_episodes=1500)

    # moving average
    moving_average_num = 100
    def moving_average(x, n=moving_average_num):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[n:] - cumsum[:-n]) / float(n)

    print(rewards)

