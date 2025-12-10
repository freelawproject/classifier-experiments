from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
from litellm import embedding
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
