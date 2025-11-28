# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 14:32:09 2025

@author: sh032
"""

import numpy as np
import pandas as pd
from paraphrase_detection import data_shuffle_split, SBERTPairClassifier, Train, TextPairDataset, log_metrics_and_model, df_tokenization
import torch
from sentence_transformers import SentenceTransformer
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import os
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

df = pd.read_parquet("hf://datasets/sentence-transformers/quora-duplicates/pair-class/train-00000-of-00001.parquet")
tokenizer=AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
encoding=tokenizer(list(df["sentence1"]), list(df["sentence2"]),padding="max_length", truncation=True, max_length=128)
print(encoding)