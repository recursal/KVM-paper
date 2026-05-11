from __future__ import annotations

import os

from dataclasses import dataclass, field
from pydantic import BaseModel, ValidationError
from typing import Any

from datasets import load_dataset

import torch.distributed as dist

class TokenizerAdapter:
    name: str

    def encode_batch_truncated(
        self, texts: list[str], max_length: int | None
    ) -> list[list[int]]:
        raise NotImplementedError


class GPT2TokenizerAdapter(TokenizerAdapter):
    def __init__(self) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding("gpt2")
        self.name = "gpt2"

    def encode_batch_truncated(
        self, texts: list[str], max_length: int | None
    ) -> list[list[int]]:
        token_batches = [self._encoding.encode_ordinary(text) for text in texts]
        if max_length is None:
            return token_batches
        return [token_ids[:max_length] for token_ids in token_batches]


class RWKVTokenizerAdapter(TokenizerAdapter):
    def __init__(self) -> None:
        import pyrwkv_tokenizer

        self._tokenizer = pyrwkv_tokenizer.RWKVTokenizer()
        self.name = "rwkv"

    def encode_batch_truncated(
        self, texts: list[str], max_length: int | None
    ) -> list[list[int]]:
        token_batches = [self._tokenizer.encode(text) for text in texts]
        if max_length is None:
            return token_batches
        return [token_ids[:max_length] for token_ids in token_batches]


class HFTokenizerAdapter(TokenizerAdapter):
    def __init__(self, name_or_path: str, trust_remote_code: bool) -> None:
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=trust_remote_code
        )
        self.name = name_or_path

    def encode_batch_truncated(
        self, texts: list[str], max_length: int | None
    ) -> list[list[int]]:
        encoded_kwargs = {
            "add_special_tokens": False,
            "return_attention_mask": False,
        }
        if max_length is None:
            encoded_kwargs["truncation"] = False
        else:
            encoded_kwargs["truncation"] = True
            encoded_kwargs["max_length"] = max_length
        encoded = self._tokenizer(texts, **encoded_kwargs)
        return encoded["input_ids"]


_TOKENIZER_CACHE: dict[tuple[str, str | None, bool, int], TokenizerAdapter] = {}


def resolve_tokenizer_mode(
    tokenizer: str,
    tokenizer_name: str | None,
    model_vocab_size: int,
) -> str:
    tokenizer_mode = str(tokenizer).lower()
    if tokenizer_mode != "auto":
        return tokenizer_mode
    if tokenizer_name:
        return "hf"
    if int(model_vocab_size) in {50_257, 50_304}:
        return "gpt2"
    if int(model_vocab_size) == 65_536:
        return "rwkv"
    raise ValueError(
        "Could not infer a tokenizer automatically. "
        "Set the tokenizer mode explicitly or provide tokenizer_name."
    )


def get_tokenizer_adapter(
    *,
    tokenizer: str,
    tokenizer_name: str | None,
    tokenizer_trust_remote_code: bool,
    model_vocab_size: int,
) -> TokenizerAdapter:
    tokenizer_mode = resolve_tokenizer_mode(tokenizer, tokenizer_name, model_vocab_size)
    cache_key = (
        tokenizer_mode,
        tokenizer_name,
        bool(tokenizer_trust_remote_code),
        int(model_vocab_size),
    )
    adapter = _TOKENIZER_CACHE.get(cache_key)
    if adapter is not None:
        return adapter

    if tokenizer_mode == "gpt2":
        adapter = GPT2TokenizerAdapter()
    elif tokenizer_mode == "rwkv":
        adapter = RWKVTokenizerAdapter()
    elif tokenizer_mode == "hf":
        adapter = HFTokenizerAdapter(
            tokenizer_name or "gpt2", bool(tokenizer_trust_remote_code)
        )
    else:
        raise ValueError(f"Unsupported tokenizer mode {tokenizer!r}.")

    _TOKENIZER_CACHE[cache_key] = adapter
    return adapter


def resolve_dataset(
    *,
    dataset: str | None,
    dataset_name: str | None,
    split: str,
):
    errors = []
    try:
        ds = load_dataset(
            dataset, name=dataset_name, split=split, streaming=False
        )
        return dataset_id, ds
    except Exception as exc:
        errors.append(f"{dataset_id}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(
        f"Failed to load dataset split {split!r} from any candidate:\n{joined}"
    )


def required_tokens(context_length: int, min_tokens: int) -> int:
    return max(int(min_tokens), int(context_length) + 1)


def map_documents(
    batch: dict[str, list[Any]],
    *,
    text_column: str,
    skip_tokens: int,
    min_tokens: int,
    context_length: int,
    tokenizer: str,
    tokenizer_name: str | None,
    tokenizer_trust_remote_code: bool,
    model_vocab_size: int,
    all_sequence_chunks: bool = False,
) -> dict[str, list[list[int] | int]]:
    if text_column not in batch:
        raise KeyError(
            f"Column {text_column!r} not found in batch. Available columns: {sorted(batch.keys())}"
        )
    if skip_tokens < 0:
        raise ValueError("skip_tokens must be non-negative.")

    tokenizer_adapter = get_tokenizer_adapter(
        tokenizer=tokenizer,
        tokenizer_name=tokenizer_name,
        tokenizer_trust_remote_code=tokenizer_trust_remote_code,
        model_vocab_size=model_vocab_size,
    )
    target_token_count = required_tokens(context_length, min_tokens)
    max_length = None if all_sequence_chunks else target_token_count + skip_tokens
    token_batches = tokenizer_adapter.encode_batch_truncated(
        batch[text_column], max_length=max_length
    )
    if skip_tokens > 0:
        token_batches = [token_ids[skip_tokens:] for token_ids in token_batches]

    output_input_ids: list[list[int]] = []
    output_labels: list[list[int]] = []
    output_token_counts: list[int] = []
    output_text_lengths: list[int] = []
    for token_ids in token_batches:
        if len(token_ids) < target_token_count:
            continue
        if all_sequence_chunks:
            max_start = len(token_ids) - context_length
            for start in range(0, max_start, context_length):
                example_tokens = token_ids[start : start + context_length + 1]
                if len(example_tokens) < context_length + 1:
                    continue
                output_input_ids.append(example_tokens[:-1])
                output_labels.append(example_tokens[1:])
                output_token_counts.append(len(token_ids))
                output_text_lengths.append(len(token_ids))
            continue

        example_tokens = token_ids[: context_length + 1]
        if len(example_tokens) < context_length + 1:
            continue
        output_input_ids.append(example_tokens[:-1])
        output_labels.append(example_tokens[1:])
        output_token_counts.append(len(token_ids))
        output_text_lengths.append(len(token_ids))

    return {
        "input_ids": output_input_ids,
        "labels": output_labels,
        "token_count": output_token_counts,
        "trimmed_text_length": output_text_lengths,
    }


def preprocess_dataset(
    dataset,
    *,
    text_column: str,
    skip_tokens: int,
    min_tokens: int,
    context_length: int,
    tokenizer: str,
    tokenizer_name: str | None,
    tokenizer_trust_remote_code: bool,
    model_vocab_size: int,
    preprocess_batch_size: int,
    desc: str,
    max_doc_count: int | None,
    seed: int | None = None,
    all_sequence_chunks: bool = False,
):
    if preprocess_batch_size <= 0:
        raise ValueError("preprocess_batch_size must be positive.")

    master_process = not dist.is_initialized() or dist.get_rank() == 0

    if dist.is_initialized() and not master_process:
        dist.barrier()

    target_token_count = required_tokens(context_length, min_tokens)

    def filter_function(example):
        return len(example[text_column]) >= target_token_count * 2.5

    num_proc = max(1, os.cpu_count() - 2)

    dataset = dataset.filter(filter_function, num_proc=num_proc)

    if seed is not None:
        dataset = dataset.shuffle(seed)

    if max_doc_count is not None:
        dataset = dataset.select(
            range(0, min(len(dataset), max_doc_count))
        )  # FIXME - temporary small subset for testing

    dataset = dataset.map(
        map_documents,
        batched=True,
        batch_size=preprocess_batch_size,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "text_column": text_column,
            "skip_tokens": skip_tokens,
            "min_tokens": min_tokens,
            "context_length": context_length,
            "tokenizer": tokenizer,
            "tokenizer_name": tokenizer_name,
            "tokenizer_trust_remote_code": bool(tokenizer_trust_remote_code),
            "model_vocab_size": int(model_vocab_size),
            "all_sequence_chunks": bool(all_sequence_chunks),
        },
        desc=desc,
        num_proc=num_proc,
    )

    if dist.is_initialized() and master_process:
        dist.barrier()

    return dataset
