import numpy as np
import glob
import random
import torch


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --train_dataset?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        print(
            "---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

    def next_shard(
        self, current_shard, current_index, generator
    ):  # advance to next data shard
        current_shard = (current_shard + 1) % len(self.files)
        current_index = 0
        self.tokens = _load_data_shard(self.files[current_shard])
        chunk_offsets = [
            (i * self.num_processes + self.process_rank) * self.T
            for i in range(len(self.tokens) // (self.T * self.num_processes))
        ]
        generator.shuffle(chunk_offsets)
        return current_shard, current_index, chunk_offsets

    def __iter__(self):
        current_shard = -1
        current_index = 0

        generator = random.Random(1234 + self.process_rank)
        current_shard, current_index, chunk_offsets = self.next_shard(
            current_shard, current_index, generator
        )
        while True:
            tensors = []
            for _ in range(self.B):
                offset = chunk_offsets[current_index]
                buf = self.tokens[offset : offset + self.T + 1]
                tensors.append(torch.tensor(buf.astype(np.int32), dtype=torch.long))
                current_index += 1
            batch_tensor = torch.stack(tensors)
            inputs = batch_tensor[:, :-1].cuda().contiguous()
            targets = batch_tensor[:, 1:].cuda().contiguous()
            # load next shard if necessary
            if current_index + self.B > len(chunk_offsets):
                current_shard, current_index, chunk_offsets = self.next_shard(
                    current_shard, current_index, generator
                )
                if current_shard == 0:
                    break
            yield dict(input_ids=inputs, labels=targets, attention_mask=None)
