from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tiktoken
from litellm import embedding
from sklearn.cluster import KMeans
from tqdm import tqdm

EMBEDDING_MODEL = "text-embedding-3-small"


def truncate_texts(texts, model=EMBEDDING_MODEL, max_tokens=8000):
    """Truncate text to a maximum number of tokens."""
    enc = tiktoken.encoding_for_model(model)
    truncated_texts = []
    for text in texts:
        tokens = enc.encode(text)
        if len(tokens) > max_tokens:
            text = enc.decode(tokens[:max_tokens])
        truncated_texts.append(text)
    return truncated_texts


def count_tokens(texts, model=EMBEDDING_MODEL):
    """Count the number of tokens in a text."""
    enc = tiktoken.encoding_for_model(model)
    return [len(enc.encode(text)) for text in texts]


def batch_embed(
    texts,
    model=EMBEDDING_MODEL,
    max_tokens=8000,
    num_workers=None,
    **embedding_args,
):
    """Batch embed texts with litellm."""
    batches = []
    token_counts = count_tokens(texts, model)
    batch_count = 0
    for i in range(len(texts)):
        text_tokens = token_counts[i]
        text = texts[i]
        if text_tokens > max_tokens:
            text = truncate_texts([text], model, max_tokens)[0]
            text_tokens = max_tokens
        if batch_count + text_tokens > max_tokens or not batches:
            batches.append([])
            batch_count = 0
        batches[-1].append(text)
        batch_count += text_tokens

    embeddings = []
    if num_workers is None:
        for batch in batches:
            r = embedding(model, batch, **embedding_args)
            embeddings.extend([x["embedding"] for x in r["data"]])
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(embedding, model, batch, **embedding_args)
                for batch in batches
            ]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
            for r in futures:
                embeddings.extend([x["embedding"] for x in r.result()["data"]])
    return embeddings


def mesh_sort(embeddings, cluster_ks):
    """Produce a diversity-maximizing ordering of embeddings via hierarchical clustering.

    This constructs a hierarchical KMeans partitioning using successive values in
    `cluster_ks`. The first K is applied globally; each subsequent K is applied
    independently within each existing cluster. Each example is assigned a
    cluster “path” across levels.

    The final ordering interleaves examples breadth-first across these paths,
    so that early indices span coarse clusters before finer subdivisions.
    Redundant or over-represented regions of the embedding space are therefore
    pushed toward the end. Ordering within identical paths is randomized.

    Returns an array of indices such that `embeddings[indices]` is diversity-ordered.
    """
    if not isinstance(cluster_ks, list) or not all(
        isinstance(x, int) for x in cluster_ks
    ):
        raise ValueError("clusters must be a list of integers")
    if not len(embeddings):
        return np.array([])
    sort = None
    for k in cluster_ks:
        if sort is None:
            use_k = min(k, len(embeddings))
            if use_k:
                sort = KMeans(n_clusters=use_k).fit_predict(embeddings)
                sort = np.vectorize(lambda x: str(x).zfill(6))(sort)
        else:
            cluster_ids = np.zeros_like(sort)
            for group in list(set(sort)):
                use_k = min(k, len(embeddings[sort == group]))
                if use_k:
                    kmeans = KMeans(n_clusters=use_k)
                    cluster_ids[sort == group] = kmeans.fit_predict(
                        embeddings[sort == group]
                    )
            cluster_ids = np.vectorize(lambda x: str(x).zfill(6))(cluster_ids)
            sort = cluster_ids + "-" + sort
    r = np.zeros_like(sort)
    for group in list(set(sort)):
        rng = np.random.default_rng()
        r[sort == group] = rng.permutation(len(sort[sort == group]))
    r = np.vectorize(lambda x: str(x).zfill(6))(r)
    sort = r + "-" + sort
    return sort.argsort()
