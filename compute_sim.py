import argparse
from gritlm import GritLM
import os
from datasets import load_dataset
import json
from retrievers import get_scores, calculate_retrieval_metrics, get_scores_v2
from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import numpy as np
import torch
import pandas as pd

tasks = [
    "biology", "earth_science", "economics", "psychology", 
    "robotics", "stackoverflow", "sustainable_living", "leetcode", 
    "pony", "aops", "theoremqa_theorems", "theoremqa_questions"
]

pre_cached_embedding_dir = "embedding_grit"

model = GritLM("GritLM/GritLM-7B", torch_dtype="auto", mode="embedding")


def main(args):
    task = args.task
    task_dir = os.path.join(pre_cached_embedding_dir, task)
    os.makedirs(task_dir, exist_ok=True)

    if args.reasoning is not None:
        split_name = f"{args.reasoning}_reason"
    else:
        split_name = "examples"
    examples = load_dataset('xlangai/bright', split_name, cache_dir='cache')[task]
    queries = []
    query_ids = []
    excluded_ids = {}
    for e in examples:
        queries.append(e["query"])
        query_ids.append(e['id'])
        excluded_ids[e['id']] = e['excluded_ids']
        overlap = set(e['excluded_ids']).intersection(set(e['gold_ids']))
        assert len(overlap)==0
    assert len(queries)==len(query_ids), f"{len(queries)}, {len(query_ids)}"

    with open(os.path.join("configs","grit",f"{args.task}.json")) as f:
        config = json.load(f)
    instructions = config['instructions']
    print(f"instructions: {instructions}")
    query_instruction = instructions['query'].format(task=task)
    doc_instruction = instructions['document']
    encode_batch_size = 64
    query_max_length = 256
    query_emb = model.encode(queries, instruction=query_instruction, batch_size=encode_batch_size, max_length=query_max_length)

    # load pre-cached document embedding embedding_grit/biology.parquet
    doc_emb_path = os.path.join(pre_cached_embedding_dir, task + ".parquet")
    doc_emb_df = pd.read_parquet(doc_emb_path)
    doc_emb = np.array(doc_emb_df['embedding'].tolist())

    print(f"query_emb.shape: {query_emb.shape}")
    print(f"doc_emb.shape: {doc_emb.shape}")
    similarity_scores = pairwise_cosine_similarity(torch.from_numpy(query_emb).to(torch.float32), torch.from_numpy(doc_emb).to(torch.float32))
    scores_v2 = get_scores_v2(query_ids=query_ids,doc_ids=doc_emb_df['id'],scores=similarity_scores,excluded_ids=excluded_ids)

    ground_truth = {}
    key = 'gold_ids'
    for e in examples:
        ground_truth[e['id']] = {}
        for gid in e[key]:
            ground_truth[e['id']][gid] = 1
        for did in e['excluded_ids']:
            assert did not in scores_v2[e['id']]
            assert not did in ground_truth[e['id']]
        # if len(ground_truth) > 20:
        #     break

    results_v2 = calculate_retrieval_metrics(scores_v2, ground_truth)
    if args.reasoning is not None:
        print(f"Grit using {args.reasoning} rewrite queries on task {task}")
    else:
        print(f"Grit using original queries on task {task}")
    print(results_v2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=tasks)
    parser.add_argument("--reasoning", type=str, default=None, choices=[
        "Gemini-1.0", "claude-3-opus", "gpt4", "grit", "llama3-70b"
    ])
    args = parser.parse_args()
    # import debugpy; debugpy.listen(5678); print("Waiting for debugger to attach...")
    # debugpy.wait_for_client()
    main(args)