from typing import Callable, Dict, Any, Optional
from copy import deepcopy

import torch
from torch.utils.data import IterableDataset

from utils.logger import print0 as print


def efficiently_tokenized(
    sharded_dataset,
    text_column,
    truncation,
    max_length_tokens,
    padding,
    accept_tokenized_document_fn,
    tokenizer,
    tokenization_batch_size=128,
):
    batch, states = [], []
    it = iter(sharded_dataset)
    done = False
    while not done:
        try:
            batch.append(next(it)[text_column])
            states.append(sharded_dataset.state_dict())
        except StopIteration:
            done = True
        if len(batch) == tokenization_batch_size or done:
            tokenizer_results = tokenizer(
                batch,
                truncation=truncation,
                max_length=max_length_tokens,
                padding=padding,
                return_attention_mask=True,
                add_special_tokens=True,
            )
            for s, tokenized, attention_mask in zip(
                states,
                tokenizer_results["input_ids"],
                tokenizer_results["attention_mask"],
            ):
                if accept_tokenized_document_fn is None or accept_tokenized_document_fn(
                    tokenized
                ):
                    yield tokenized, attention_mask, s
            batch, states = [], []


class TokenizingDatasetParent(IterableDataset):
    def __init__(
        self,
        iterable,
        text_column,
        rank,
        world_size,
        sequence_len,
        max_doc_tokens,
        accept_tokenized_document_fn,
        tokenizer,
        tokenization_batch_size=128,
    ):
        self.iterable = iterable
        self.text_column = text_column

        self.rank = rank
        self.world_size = world_size
        self.sequence_len_plus_one = (
            sequence_len + 1
        )  # NOTE - this is so that collator can obtain properly size input_ids and labels
        self.max_doc_tokens_plus_one = (
            max_doc_tokens + 1
        )  # NOTE - this is so that collator can obtain properly size input_ids and labels
        self.tokenizer = tokenizer
        self.tokenization_batch_size = tokenization_batch_size
        self.accept_tokenized_document_fn = accept_tokenized_document_fn

        self.states = None

    def create_iterable_sharded_dataset(self):
        # this is counterintuitive - it chooses a sub shard of the total shards which are world_size * num_workers, so that each worker gets a unique subset of the data
        return self.iterable.shard(num_shards=self.world_size, index=self.rank)

    def state_dict(self):
        return dict(states=self.states)

    def load_state_dict(self, state_dict):
        self.states = state_dict["states"]


class TokenizingDataset(TokenizingDatasetParent):
    def __init__(
        self,
        iterable,
        text_column,
        rank,
        world_size,
        sequence_len,
        max_doc_tokens,
        accept_tokenized_document_fn,
        tokenizer,
        padding="max_length",
        tokenization_batch_size=128,
    ):
        super().__init__(
            iterable,
            text_column,
            rank,
            world_size,
            sequence_len,
            max_doc_tokens,
            accept_tokenized_document_fn,
            tokenizer,
            tokenization_batch_size,
        )
        self.padding = padding

    def __iter__(self):
        sharded_dataset = self.create_iterable_sharded_dataset()

        if self.states is not None:
            sharded_dataset.load_state_dict(self.states)

        for tokenized_doc, attention_mask, states in efficiently_tokenized(
            sharded_dataset=sharded_dataset,
            text_column=self.text_column,
            truncation=True,
            max_length_tokens=min(
                self.sequence_len_plus_one, self.max_doc_tokens_plus_one
            ),
            padding=self.padding,
            accept_tokenized_document_fn=self.accept_tokenized_document_fn,
            tokenizer=self.tokenizer,
            tokenization_batch_size=self.tokenization_batch_size,
        ):
            self.states = states

            input_ids = torch.tensor(tokenized_doc[:-1], dtype=torch.long)
            attention_mask = torch.tensor(attention_mask[:-1], dtype=torch.int)
            labels = torch.tensor(tokenized_doc[1:], dtype=torch.long)
            labels[attention_mask == 0] = -100
            yield dict(input_ids=input_ids, labels=labels)


class BufferPackedTokenizingDataset(TokenizingDatasetParent):
    def __init__(
        self,
        iterable,
        text_column,
        rank,
        world_size,
        sequence_len,
        max_doc_tokens,
        accept_tokenized_document_fn,
        tokenizer,
        tokenization_batch_size=128,
        buffer_num_sequences=16,
    ):
        super().__init__(
            iterable,
            text_column,
            rank,
            world_size,
            sequence_len,
            max_doc_tokens,
            accept_tokenized_document_fn,
            tokenizer,
            tokenization_batch_size,
        )

        self.buffer_num_sequences = buffer_num_sequences
        self.epoch = 0

        self.buffer = []
        self.rng_state = None

    def __iter__(self):
        sharded_dataset = self.create_iterable_sharded_dataset()

        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank)
        if self.rng_state is not None:
            g.set_state(self.rng_state)

        for tokenized_doc, attention_mask, states in efficiently_tokenized(
            sharded_dataset=sharded_dataset,
            text_column=self.text_column,
            truncation=False,
            max_length_tokens=None,
            padding="do_not_pad",
            accept_tokenized_document_fn=self.accept_tokenized_document_fn,
            tokenizer=self.tokenizer,
            tokenization_batch_size=self.tokenization_batch_size,
        ):
            # fill the buffer until it's full enough
            self.buffer += tokenized_doc
            self.states = states

            # usually skipped, this drains the buffer once it's full
            if (
                len(self.buffer)
                >= self.buffer_num_sequences * self.sequence_len_plus_one
            ):
                yield from self.drain_buffer(g)

        # drain any remaining buffer at the end of the epoch
        yield from self.drain_buffer(g)

    def drain_buffer(self, g):
        n_sequences = len(self.buffer) // self.sequence_len_plus_one
        if n_sequences > 0:
            n_tokens = n_sequences * self.sequence_len_plus_one
            print(
                "draining buffer",
                len(self.buffer),
                n_tokens,
                self.sequence_len_plus_one,
                n_sequences,
                len(self.buffer[n_tokens:]),
            )
            sequences = torch.tensor(self.buffer[:n_tokens], dtype=torch.long).view(
                n_sequences, -1
            )
            for i in torch.randperm(n_sequences, generator=g).tolist():
                yield {"input_ids": sequences[i, :-1], "labels": sequences[i, 1:]}
            self.rng_state = g.get_state()
            self.buffer = self.buffer[n_tokens:]

    def state_dict(self):
        return dict(
            states=self.states,
            buffer=deepcopy(self.buffer),
            rng_state=deepcopy(self.rng_state),
        )

    def load_state_dict(self, state_dict):
        self.states = state_dict["states"]
        self.buffer = deepcopy(state_dict["buffer"])
        self.rng_state = deepcopy(state_dict["rng_state"])


class BufferSmartPackedTokenizingDataset(TokenizingDatasetParent):
    def __init__(
        self,
        iterable,
        text_column,
        rank,
        world_size,
        sequence_len,
        max_doc_tokens,
        accept_tokenized_document_fn,
        tokenizer,
        tokenization_batch_size=128,
        buffer_num_sequences=64,
    ):
        super().__init__(
            iterable,
            text_column,
            rank,
            world_size,
            sequence_len,
            max_doc_tokens,
            accept_tokenized_document_fn,
            tokenizer,
            tokenization_batch_size,
        )

        self.buffer_num_sequences = buffer_num_sequences
        self.epoch = 0

        self.tokenized_amalgam = []
        self.buffer = []
        self.rng_state = None

    def __iter__(self):
        sharded_dataset = self.create_iterable_sharded_dataset()

        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank)
        if self.rng_state is not None:
            g.set_state(self.rng_state)

        for tokenized_doc, attention_mask, states in efficiently_tokenized(
            sharded_dataset=sharded_dataset,
            text_column=self.text_column,
            truncation=False,
            max_length_tokens=None,
            padding="do_not_pad",
            accept_tokenized_document_fn=self.accept_tokenized_document_fn,
            tokenizer=self.tokenizer,
            tokenization_batch_size=self.tokenization_batch_size,
        ):
            # split up tokenized_doc into sub documents that fit seqlen
            if len(tokenized_doc) < self.sequence_len_plus_one:
                self.tokenized_amalgam += tokenized_doc
                if len(self.tokenized_amalgam) >= self.sequence_len_plus_one:
                    self.buffer += self.tokenized_amalgam[: self.sequence_len_plus_one]
                    self.tokenized_amalgam = []  # self.tokenized_amalgam[self.sequence_len_plus_one:]
            elif len(tokenized_doc) >= self.sequence_len_plus_one:
                self.buffer += tokenized_doc[: self.sequence_len_plus_one]
                for i in range(
                    self.sequence_len_plus_one,
                    len(tokenized_doc) - self.sequence_len_plus_one,
                    self.sequence_len_plus_one,
                ):
                    self.buffer += tokenized_doc[i : i + self.sequence_len_plus_one]
                # for now, drop anything that doesn't fit!
                # if len(tokenized_doc) % self.sequence_len_plus_one != 0:
                #    self.tokenized_docs += [tokenized_doc[len(tokenized_doc) - (len(tokenized_doc) % self.sequence_len_plus_one):]]

            self.states = states
            if (
                len(self.buffer)
                >= self.buffer_num_sequences * self.sequence_len_plus_one
            ):
                yield from self.drain_buffer(g)

        # drain any remaining buffer at the end of the epoch
        yield from self.drain_buffer(g)

    def drain_buffer(self, g):
        n_sequences = len(self.buffer) // self.sequence_len_plus_one
        if n_sequences > 0:
            n_tokens = n_sequences * self.sequence_len_plus_one
            print(
                "draining buffer",
                len(self.buffer),
                n_tokens,
                self.sequence_len_plus_one,
                n_sequences,
                len(self.buffer[n_tokens:]),
            )
            sequences = torch.tensor(self.buffer[:n_tokens], dtype=torch.long).view(
                n_sequences, -1
            )
            for i in torch.randperm(n_sequences, generator=g).tolist():
                yield {"input_ids": sequences[i, :-1], "labels": sequences[i, 1:]}
            self.rng_state = g.get_state()
            self.buffer = self.buffer[n_tokens:]

    def state_dict(self):
        return dict(
            states=self.states,
            buffer=deepcopy(self.buffer),
            rng_state=deepcopy(self.rng_state),
        )

    def load_state_dict(self, state_dict):
        self.states = state_dict["states"]
        self.buffer = deepcopy(state_dict["buffer"])
        self.rng_state = deepcopy(state_dict["rng_state"])


class CrammedTokenizingDataset(TokenizingDatasetParent):
    def __init__(
        self,
        iterable,
        text_column,
        rank,
        world_size,
        sequence_len,
        max_doc_tokens,
        accept_tokenized_document_fn,
        tokenizer,
        tokenization_batch_size=128,
    ):
        super().__init__(
            iterable,
            text_column,
            rank,
            world_size,
            sequence_len,
            max_doc_tokens,
            accept_tokenized_document_fn,
            tokenizer,
            tokenization_batch_size,
        )

        self.tokenized_amalgam = []

    def __iter__(self):
        sharded_dataset = self.create_iterable_sharded_dataset()

        if self.states is not None:
            sharded_dataset.load_state_dict(self.states)

        for tokenized_doc, attention_mask, states in efficiently_tokenized(
            sharded_dataset=sharded_dataset,
            text_column=self.text_column,
            truncation=True,
            max_length_tokens=min(
                self.sequence_len_plus_one, self.max_doc_tokens_plus_one - 1
            ),  # NOTE - using -1 here because otherwise it can exceed limits
            padding="do_not_pad",
            accept_tokenized_document_fn=self.accept_tokenized_document_fn,
            tokenizer=self.tokenizer,
            tokenization_batch_size=self.tokenization_batch_size,
        ):
            self.states = states

            # if doc is seqlen/maxdoc, add directly to buffer, otherwise add to amalgam to be packed with other docs
            if len(tokenized_doc) == self.sequence_len_plus_one:
                yield self.emit(tokenized_doc)
            else:
                self.tokenized_amalgam += tokenized_doc
                if len(self.tokenized_amalgam) >= self.sequence_len_plus_one:
                    yield self.emit(
                        self.tokenized_amalgam[: self.sequence_len_plus_one]
                    )
                    self.tokenized_amalgam = []  # self.tokenized_amalgam[self.sequence_len_plus_one:] # for now, discard the remaining amalgam
                self.states = states

    def emit(self, packed_tokens):
        packed_tokens = packed_tokens[: self.sequence_len_plus_one]

        input_ids = torch.tensor(packed_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(packed_tokens[1:], dtype=torch.long)
        # prevent label leakage across doc boundaries by setting labels to -100 at EOS token positions
        labels[input_ids == self.tokenizer.eos_token_id] = -100

        return dict(input_ids=input_ids, labels=labels)

    def state_dict(self):
        return dict(
            states=self.states, tokenized_amalgam=deepcopy(self.tokenized_amalgam)
        )

    def load_state_dict(self, state_dict):
        self.states = state_dict["states"]
        self.tokenized_amalgam = deepcopy(state_dict["tokenized_amalgam"])


class PackedTokenizingDataset(TokenizingDatasetParent):
    # FIXME: cu_seqlens is built as document starts, but conventional cumulative sequence lengths usually also include the final end offset. Some model code appears to expect a true boundary list ending at total length.
    def __init__(
        self,
        iterable,
        text_column,
        rank,
        world_size,
        sequence_len,
        max_doc_tokens,
        accept_tokenized_document_fn,
        tokenizer,
        return_cu_seqlens,
        sorted,
        tokenization_batch_size=128,
    ):
        super().__init__(
            iterable,
            text_column,
            rank,
            world_size,
            sequence_len,
            max_doc_tokens,
            accept_tokenized_document_fn,
            tokenizer,
            tokenization_batch_size,
        )
        self.return_cu_seqlens = return_cu_seqlens

        self.tokenized_docs = []

        self.sorted = sorted

    def __iter__(self):
        sharded_dataset = self.create_iterable_sharded_dataset()

        if self.states is not None:
            sharded_dataset.load_state_dict(self.states)

        packed_len = sum(len(tokenized_doc) for tokenized_doc in self.tokenized_docs)
        for tokenized_doc, attention_mask, states in efficiently_tokenized(
            sharded_dataset=sharded_dataset,
            text_column=self.text_column,
            truncation=True,
            max_length_tokens=min(
                self.sequence_len_plus_one, self.max_doc_tokens_plus_one - 1
            ),  # NOTE - using -1 here because otherwise it can exceed limits
            padding="do_not_pad",
            accept_tokenized_document_fn=self.accept_tokenized_document_fn,
            tokenizer=self.tokenizer,
            tokenization_batch_size=self.tokenization_batch_size,
        ):
            self.tokenized_docs += [tokenized_doc]
            packed_len += len(tokenized_doc)
            self.states = states

            # usually skipped, this only emits a set of packed tokens if there are enough to fill the sequence length
            while packed_len >= self.sequence_len_plus_one:
                if self.sorted:
                    # sort the tokenized documents by length and pack them in that order to minimize padding and improve ctx usage
                    self.tokenized_docs.sort(key=len, reverse=True)

                cu_seqlens = []
                packed_tokens = []
                for tokenized_doc in self.tokenized_docs:
                    cu_seqlens += [len(packed_tokens)]
                    packed_tokens += tokenized_doc

                yield self.emit(packed_tokens, cu_seqlens)
                self.tokenized_docs = []
                packed_len = 0

    def emit(self, packed_tokens, cu_seqlens):
        packed_tokens = packed_tokens[: self.sequence_len_plus_one]
        cu_seqlens = cu_seqlens[: self.sequence_len_plus_one - 1]

        input_ids = torch.tensor(packed_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(packed_tokens[1:], dtype=torch.long)
        # prevent label leakage across doc boundaries
        # FIXME - might really be better to check for EOS instead of cu_seqlen
        for doc_start in cu_seqlens[1:]:
            if doc_start <= self.sequence_len_plus_one - 1:
                labels[doc_start - 1] = -100

        rv = dict(input_ids=input_ids, labels=labels)
        if self.return_cu_seqlens:
            rv["cu_seqlens"] = cu_seqlens
        return rv

    def state_dict(self):
        return dict(states=self.states, tokenized_docs=deepcopy(self.tokenized_docs))

    def load_state_dict(self, state_dict):
        self.states = state_dict["states"]
        self.tokenized_docs = deepcopy(state_dict["tokenized_docs"])


import pickle
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader

# def make_worker_init_fn(process_id, num_workers, total_shards):
#     def worker_init_fn(worker_id):
#         worker_info = torch.utils.data.get_worker_info()
#         dataset = worker_info.dataset
#         global_worker_shard_id = process_id * num_workers + worker_id
#         dataset.iterable = dataset.iterable.shard(
#             num_shards=total_shards,
#             index=global_worker_shard_id
#         )
#     return worker_init_fn


class ParallelAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        dataset: IterableDataset,
        batch_size: int,
        collate_fn: Callable | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        snapshot_every_n_steps: Optional[int] = 1,
    ):
        total_shards = num_workers * world_size
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            snapshot_every_n_steps=snapshot_every_n_steps,
            # worker_init_fn=make_worker_init_fn(rank, num_workers, total_shards)
        )
        self.rank = rank

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {f"rank_{self.rank}": pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid
        if not state_dict:
            return

        if f"rank_{self.rank}" not in state_dict:
            # FIXME - need logger
            # logger.warning(f'DataLoader state is empty for dp rank {self.rank}, expected key rank_{self.rank}')
            return
        super().load_state_dict(pickle.loads(state_dict[f"rank_{self.rank}"]))
