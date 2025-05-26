from datasets import load_dataset
import os
import torch
import pandas as pd
from gritlm import GritLM
import json

tasks = [
    "biology", "earth_science", "economics", "psychology", 
    "robotics", "stackoverflow", "sustainable_living", "leetcode", 
    "pony", "aops", "theoremqa_theorems", "theoremqa_questions"
]
config_dir = "configs/grit"

embedding_dir = "embedding_grit"
os.makedirs(embedding_dir, exist_ok=True)
encode_batch_size = 32

model = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")
model.eval()

for task in tasks:
    # doc_pairs = load_dataset('xlangai/bright', 'long_documents',cache_dir=args.cache_dir)[args.task]
    print(f"Caching document embedding for {task}")
    doc_pairs = load_dataset('xlangai/bright', 'documents', cache_dir='cache')[task]
    doc_pair_df = pd.DataFrame(doc_pairs)
    # columns: id, content

    config_file = os.path.join(config_dir, task + ".json")
    config = json.load(open(config_file))
    doc_instruction = config['instructions']['document']

    doc_max_length = 2048
    doc_emb = model.encode(doc_pair_df['content'], instruction=doc_instruction, batch_size=encode_batch_size, max_length=doc_max_length)
    
    doc_pair_df['embedding'] = doc_emb.tolist()
    doc_pair_df.to_parquet(os.path.join(embedding_dir, f"{task}.parquet"))
    print(f"Cached document embedding for {task}")
    print()