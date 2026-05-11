from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VarlenBucketPlan:
    chunk_size: int
    padded_len: int
    batch_size: int
    gather_indices: torch.Tensor
    valid_output_positions: torch.Tensor
    output_positions: torch.Tensor


def build_bucket_plans(
    cu_seqlens: list[int] | None,
    device="cpu",
    chunk_size: int = 1,
    bucket_seqlens=[256, 512, 1024, 2048, 4096],
) -> list[VarlenBucketPlan] | None:
    if cu_seqlens is None:
        return None

    def chunk_ceil(x):
        return (x + chunk_size - 1) // chunk_size * chunk_size

    # if chunked, switch to chunk based positions and lengths instead of token based
    if chunk_size > 1:
        assert all(seqlen % chunk_size == 0 for seqlen in cu_seqlens), (
            "cu_seqlens were not all exact multiples of chunk_size"
        )
        cu_seqlens = [chunk_ceil(seqlen) // chunk_size for seqlen in cu_seqlens]
        assert all(seqlen % chunk_size == 0 for seqlen in bucket_seqlens), (
            "bucket_seqlens were not all exact multiples of chunk_size"
        )
        bucket_seqlens = [chunk_ceil(seqlen) // chunk_size for seqlen in bucket_seqlens]

    cu_seqlens_cpu = torch.tensor(cu_seqlens, dtype=torch.long, device="cpu")
    # cu_seqlens_cpu = cu_seqlens.detach().to(device='cpu', dtype=torch.long)
    if cu_seqlens_cpu.ndim != 1 or cu_seqlens_cpu.shape[0] < 2:
        raise ValueError(
            f"cu_seqlens must be 1D with at least two entries, got shape {tuple(cu_seqlens_cpu.shape)}"
        )

    total_length = int(cu_seqlens_cpu[-1].item())
    doc_starts = cu_seqlens_cpu[:-1].tolist()
    doc_ends = cu_seqlens_cpu[1:].tolist()

    bucket_entries: dict[int, list[tuple[int, int, int]]] = {}
    for doc_idx, (start, end) in enumerate(zip(doc_starts, doc_ends)):
        doc_len = end - start
        if doc_len < 0:
            raise ValueError("cu_seqlens must be nondecreasing")
        # padded_len = self.chunk_ceil(doc_len)
        padded_len = None
        for bucket_seqlen in bucket_seqlens:
            if doc_len <= bucket_seqlen:
                padded_len = bucket_seqlen
                break
        if padded_len is None:
            raise ValueError(
                f"document length {doc_len} exceeds the largest configured bucket size {max(bucket_seqlens)}"
            )
        bucket_entries.setdefault(padded_len, []).append((doc_idx, start, doc_len))

    bucket_plans = []
    pad_index = 0  # total_length # NOTE - changed this so we don't need padding
    for padded_len, entries in bucket_entries.items():
        gather_indices: list[int] = []
        valid_output_positions: list[int] = []
        output_positions: list[int] = []
        for row_idx, (_, start, doc_len) in enumerate(entries):
            gather_indices.extend(range(start, start + doc_len))
            gather_indices.extend([pad_index] * (padded_len - doc_len))
            valid_output_positions.extend(
                row_idx * padded_len + offset for offset in range(doc_len)
            )
            output_positions.extend(range(start, start + doc_len))

        bucket_plans.append(
            VarlenBucketPlan(
                chunk_size=chunk_size,
                padded_len=padded_len,
                batch_size=len(entries),
                gather_indices=torch.tensor(
                    gather_indices, dtype=torch.long, device=device
                ),
                valid_output_positions=torch.tensor(
                    valid_output_positions, dtype=torch.long, device=device
                ),
                output_positions=torch.tensor(
                    output_positions, dtype=torch.long, device=device
                ),
            )
        )

    return bucket_plans


# def gather_bucket_tensor(flat_tensor: torch.Tensor, plan: VarlenBucketPlan) -> torch.Tensor:
#     padded_source = torch.cat([flat_tensor, flat_tensor.new_zeros((1, *flat_tensor.shape[1:]))], dim=0)
#     gathered = padded_source.index_select(0, plan.gather_indices)
#     return gathered.view(plan.batch_size, plan.padded_len, *flat_tensor.shape[1:])


def gather_bucket_tensor_stack(
    flat_tensor: torch.Tensor, plan: VarlenBucketPlan
) -> torch.Tensor:
    # NOTE - changed this so we don't need padding, by assuming that the last position of the input tensor is a pad token that can be safely gathered for padding purposes
    # padded_source = torch.cat([flat_tensor, flat_tensor.new_zeros((flat_tensor.shape[0], 1, *flat_tensor.shape[2:]))], dim=1)
    padded_source = flat_tensor.view(
        flat_tensor.shape[0], -1, plan.chunk_size, *flat_tensor.shape[2:]
    )
    gathered = padded_source.index_select(1, plan.gather_indices)
    return gathered.view(
        flat_tensor.shape[0],
        plan.batch_size,
        plan.padded_len * plan.chunk_size,
        *flat_tensor.shape[2:],
    )


def bucket_tensors_and_call_fn(tensors, bucket_plans, fn):
    if bucket_plans is None:
        return fn(*tensors)

    gathered_bucket_outputs = []
    gathered_output_positions = []

    # tensors come with a batch dimension of size 1
    for tensor in tensors:
        if tensor.shape[0] != 1:
            raise ValueError(
                f"Expected tensors to have batch dimension of size 1, got shape {tuple(tensor.shape)}"
            )
    stacked_tensors = torch.cat(tensors, dim=0)
    for plan in bucket_plans:
        stacked_batch = gather_bucket_tensor_stack(stacked_tensors, plan)
        batch = stacked_batch.unbind(dim=0)
        # NOTE - now these have no batch dim

        batch_output = fn(*batch)
        flat_bucket_output = batch_output.reshape(
            plan.batch_size * plan.padded_len, plan.chunk_size, *batch_output.shape[2:]
        )
        valid_bucket_output = flat_bucket_output.index_select(
            0,
            plan.valid_output_positions,
        )
        gathered_bucket_outputs.append(valid_bucket_output)
        gathered_output_positions.append(plan.output_positions)

    flat_output = torch.cat(gathered_bucket_outputs, dim=0)
    output_positions = torch.cat(gathered_output_positions, dim=0)
    # if output_positions.numel() != total_tokens:
    #     raise RuntimeError(
    #         f'Reassembled varlen output collected {output_positions.numel()} token positions, expected {total_tokens}'
    #     )
    reorder = torch.argsort(output_positions)
    flat_output = flat_output.index_select(0, reorder)
    flat_output = flat_output.view(1, -1, *flat_output.shape[2:])
    return flat_output


def pad_documents_to_chunk_size(
    tokens: torch.Tensor,
    cu_seqlens: list[int] | torch.Tensor,
    chunk_size: int,
) -> tuple[torch.Tensor, list[int]]:
    """Round each document length up to a multiple of chunk_size by appending zeros.

    Args:
        tokens: Flat token tensor of shape [total_tokens, ...].
        cu_seqlens: Cumulative sequence lengths (length = num_docs + 1).
        chunk_size: Pad each document to the next multiple of this value.

    Returns:
        (padded_tokens, new_cu_seqlens) where padded_tokens has shape
        [new_total_tokens, ...] and new_cu_seqlens is a list[int].
    """
    if isinstance(cu_seqlens, torch.Tensor):
        cu_list: list[int] = cu_seqlens.tolist()
    else:
        cu_list = list(cu_seqlens)

    chunks: list[torch.Tensor] = []
    new_cu: list[int] = [0]
    running = 0
    for start, end in zip(cu_list[:-1], cu_list[1:]):
        doc_len = end - start
        padded_len = ((doc_len + chunk_size - 1) // chunk_size) * chunk_size
        chunks.append(tokens[start:end])
        if padded_len > doc_len:
            chunks.append(tokens.new_zeros((padded_len - doc_len, *tokens.shape[1:])))
        running += padded_len
        new_cu.append(running)

    if not chunks:
        return tokens.new_zeros((0, *tokens.shape[1:])), new_cu

    return torch.cat(chunks, dim=0), new_cu


def unpad_documents(
    padded_tokens: torch.Tensor,
    old_cu_seqlens: list[int],
    new_cu_seqlens: list[int],
) -> torch.Tensor:
    """Undo pad_documents_to_chunk_size, recovering the original token tensor.

    Args:
        padded_tokens: Flat token tensor returned by pad_documents_to_chunk_size,
            shape [new_total_tokens, ...].
        old_cu_seqlens: Original cumulative sequence lengths before padding.
        new_cu_seqlens: Cumulative sequence lengths after padding (as returned by
            pad_documents_to_chunk_size).

    Returns:
        Token tensor of shape [old_total_tokens, ...] matching the original input.
    """
    chunks: list[torch.Tensor] = []
    for i in range(len(old_cu_seqlens) - 1):
        old_doc_len = old_cu_seqlens[i + 1] - old_cu_seqlens[i]
        new_start = new_cu_seqlens[i]
        chunks.append(padded_tokens[new_start : new_start + old_doc_len])

    if not chunks:
        return padded_tokens.new_zeros((0, *padded_tokens.shape[1:]))

    return torch.cat(chunks, dim=0)
