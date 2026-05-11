from typing import Any, Dict, List, Optional, Tuple, Union

import os
import torch
from datasets import load_dataset
from dataclasses import dataclass

from transformers import AutoTokenizer, PreTrainedTokenizer

from train import DatasetDetails

from data.varlen_dataset import (
    TokenizingDataset,
    CrammedTokenizingDataset,
    BufferPackedTokenizingDataset,
    BufferSmartPackedTokenizingDataset,
    PackedTokenizingDataset,
    ParallelAwareDataLoader,
)


def preprocess_dataset_and_get_dataloader(
    ddp_rank,
    ddp_world_size,
    args: DatasetDetails,
    tokenizer: PreTrainedTokenizer,
    preprocess_batch_size: int,
):
    master_process = ddp_rank == 0

    assert args.data_packing in [
        "varlen",
        "cram",
        "bufferpack",
        "buffersmartpack",
        "pack",
        "sortpack",
        "prepacked",
        "1",
        "pad",
        "0",
    ]

    if args.data_packing in {"prepacked", "1"}:
        from data.speedrun_dataloader import DistributedDataLoader

        data_loader = DistributedDataLoader(
            args.dataset,
            args.device_batch_size,
            args.sequence_length,
            ddp_rank,
            ddp_world_size,
        )
        if master_process:
            print(
                f"DataLoader: total number of tokens: {data_loader.ntok_total} across {len(data_loader.files)} files"
            )
        return data_loader

    # data is not prepacked or tokenized

    if not master_process:
        torch.distributed.barrier()

    dataset = load_dataset(
        args.dataset, name=args.dataset_name, split=args.split, streaming=False
    )

    if preprocess_batch_size <= 0:
        raise ValueError("train.preprocess_batch_size must be positive.")

    if args.min_document_chars is not None and args.min_document_chars > 0:

        def filter_function(example):
            return len(example[args.text_column]) >= args.min_document_chars

        dataset = dataset.filter(filter_function, num_proc=max(1, os.cpu_count() - 2))

    if args.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=args.shuffle_seed)

    if args.range_begin is not None or args.range_end is not None:
        start = args.range_begin or 0
        end = args.range_end or len(dataset)
        dataset = dataset.select(range(start, end))

    if master_process:
        torch.distributed.barrier()

    if args.min_document_tokens is not None and args.min_document_tokens > 0:
        accept_tokenized_document_fn = lambda tokenized: (
            len(tokenized) >= args.min_document_tokens
        )
    else:
        accept_tokenized_document_fn = None

    num_workers = 4
    total_shards = num_workers * ddp_world_size

    if args.data_packing == "pad_og":

        @dataclass
        class TruncatingTokenizingCollator:
            tokenizer: AutoTokenizer
            max_length: int

            def __call__(
                self, examples: List[Dict[str, Union[str, List[str]]]]
            ) -> Dict[str, torch.Tensor]:
                texts = [example["text"] for example in examples]
                tokens = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.max_length + 1,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"][:, :-1]
                attention_mask = tokens["attention_mask"][:, :-1]
                labels = tokens["input_ids"][:, 1:].clone()
                labels[attention_mask == 0] = -100
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }

        from torch.utils.data.distributed import DistributedSampler
        from torch.utils.data import DataLoader

        data_loader = DataLoader(
            dataset,
            batch_size=args.device_batch_size,
            collate_fn=TruncatingTokenizingCollator(
                tokenizer=tokenizer,
                max_length=min(args.max_document_tokens, args.sequence_length),
            ),
            num_workers=4,
            pin_memory=True,  # this was causing problems here somehow, unclear why that started happening
            sampler=DistributedSampler(dataset, drop_last=True),
        )
        return data_loader

    dataset = dataset.to_iterable_dataset(num_shards=total_shards)

    # # Use HF's native distributed sharding instead of manual .shard()
    # dataset = dataset.distribute(
    #     rank=ddp_rank,
    #     world_size=ddp_world_size,
    # )

    if args.data_packing == "varlen":
        dataset = PackedTokenizingDataset(
            dataset,
            text_column=args.text_column,
            rank=ddp_rank,
            world_size=ddp_world_size,
            sequence_len=args.sequence_length,
            max_doc_tokens=args.max_document_tokens,
            accept_tokenized_document_fn=accept_tokenized_document_fn,
            tokenizer=tokenizer,
            return_cu_seqlens=True,
            sorted=False,
            tokenization_batch_size=preprocess_batch_size,
        )
        assert args.device_batch_size == 1, (
            "for varlen data packing we require train device batch size to be 1"
        )
    elif args.data_packing == "bufferpack":
        dataset = BufferPackedTokenizingDataset(
            dataset,
            text_column=args.text_column,
            rank=ddp_rank,
            world_size=ddp_world_size,
            sequence_len=args.sequence_length,
            max_doc_tokens=args.max_document_tokens,
            accept_tokenized_document_fn=accept_tokenized_document_fn,
            tokenizer=tokenizer,
            tokenization_batch_size=preprocess_batch_size,
        )
    elif args.data_packing == "buffersmartpack":
        dataset = BufferSmartPackedTokenizingDataset(
            dataset,
            text_column=args.text_column,
            rank=ddp_rank,
            world_size=ddp_world_size,
            sequence_len=args.sequence_length,
            max_doc_tokens=args.max_document_tokens,
            accept_tokenized_document_fn=accept_tokenized_document_fn,
            tokenizer=tokenizer,
            tokenization_batch_size=preprocess_batch_size,
        )
    elif args.data_packing == "cram":
        dataset = CrammedTokenizingDataset(
            dataset,
            text_column=args.text_column,
            rank=ddp_rank,
            world_size=ddp_world_size,
            sequence_len=args.sequence_length,
            max_doc_tokens=args.max_document_tokens,
            accept_tokenized_document_fn=accept_tokenized_document_fn,
            tokenizer=tokenizer,
            tokenization_batch_size=preprocess_batch_size,
        )
    elif args.data_packing in ["pack", "sortpack"]:
        sorted = args.data_packing == "sortpack"
        dataset = PackedTokenizingDataset(
            dataset,
            text_column=args.text_column,
            rank=ddp_rank,
            world_size=ddp_world_size,
            sequence_len=args.sequence_length,
            max_doc_tokens=args.max_document_tokens,
            accept_tokenized_document_fn=accept_tokenized_document_fn,
            tokenizer=tokenizer,
            return_cu_seqlens=False,
            sorted=sorted,
            tokenization_batch_size=preprocess_batch_size,
        )
    elif args.data_packing in {"pad", "0"}:
        dataset = TokenizingDataset(
            dataset,
            text_column=args.text_column,
            rank=ddp_rank,
            world_size=ddp_world_size,
            sequence_len=args.sequence_length,
            max_doc_tokens=args.max_document_tokens,
            accept_tokenized_document_fn=accept_tokenized_document_fn,
            tokenizer=tokenizer,
            tokenization_batch_size=preprocess_batch_size,
        )

    data_loader = ParallelAwareDataLoader(
        dataset=dataset,
        batch_size=args.device_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        rank=ddp_rank,
        world_size=ddp_world_size,
        # collate_fn=collate_fn,
    )

    return data_loader
