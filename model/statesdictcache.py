import torch
from typing import Any

from transformers.cache_utils import Cache as HuggingFaceCache, CacheLayerMixin


class StatesDictLayer(CacheLayerMixin):
    is_compileable = True
    is_sliding = False

    def __init__(self):
        super().__init__()
        self.reset()

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor):
        self.reset()

    def update(
        self,
        *,
        states_dict: dict[str, Any] | None = None,
        offset: int,
        **_: Any,
    ) -> dict[str, Any]:
        new_states = states_dict
        del states_dict
        if new_states is not None:
            dynamic_state_names = ["k", "v", "swa_k", "swa_v", "bswa_k", "bswa_v"]
            for dict_key, new_entry in new_states.items():
                if dict_key not in dynamic_state_names:
                    self.states_dict[dict_key] = new_entry

            swa_size = self.states_dict.get("swa_size", 1024)
            sink_size = self.states_dict.get("sink_size", 1)
            bswa_block_size = self.states_dict.get("bswa_block_size", 256)
            bswa_n_blocks = self.states_dict.get("bswa_n_blocks", 2)

            for dict_key, new_entry in new_states.items():
                if dict_key in dynamic_state_names:
                    if dict_key not in self.states_dict:
                        self.states_dict[dict_key] = new_entry
                    else:
                        old_entry = self.states_dict[dict_key]
                        if dict_key in ["k", "v"]:
                            out = torch.cat([old_entry, new_entry], dim=1)

                        if dict_key in ["swa_k", "swa_v"]:
                            out = torch.cat([old_entry, new_entry], dim=1)
                            if sink_size == 0:
                                out = out[:, -swa_size:]
                            else:
                                out = torch.cat(
                                    [out[:, :sink_size], out[:, -swa_size:]], dim=1
                                )

                        if dict_key in ["bswa_k", "bswa_v"]:
                            out = torch.cat([old_entry, new_entry], dim=1)
                            new_seen_tokens = self._seen_tokens + new_entry.shape[1]
                            new_seen_tokens_ceil = (
                                (new_seen_tokens + bswa_block_size - 1)
                                // bswa_block_size
                                * bswa_block_size
                            )
                            new_seen_tokens_bswa_begin = max(
                                0,
                                new_seen_tokens_ceil - bswa_n_blocks * bswa_block_size,
                            )
                            amt_to_keep = max(
                                0, new_seen_tokens - new_seen_tokens_bswa_begin
                            )
                            kept = out[:, -amt_to_keep:]
                            if sink_size == 0:
                                out = kept
                            else:
                                if out.shape[1] > amt_to_keep + sink_size:
                                    out = torch.cat([out[:, :sink_size], kept], dim=1)

                        self.states_dict[dict_key] = out

        self._seen_tokens += offset
        return self.states_dict

    def reset(self):
        self.states_dict = {}
        self._seen_tokens = 0

    def get_seq_length(self) -> int:
        # Returns the sequence length of the cache for the given layer.
        return self._seen_tokens

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        # Return a tuple (kv_length, kv_offset) corresponding to the length and offset that will be returned for the given layer at layer_idx. The masks are then prepared according to the given lengths (kv_length, kv_offset) and patterns for each layer.
        return 0, 0

    def get_max_cache_shape(self) -> int:
        # Returns maximum sequence length of the cache object. Dynamic caches do not have a maximum length.
        return -1

    def offload(self):
        def to_cpu(x):
            return x.to("cpu", non_blocking=True) if isinstance(x, torch.Tensor) else x

        for k, v in self.states_dict.items():
            if v is None:
                continue
            if isinstance(v, (tuple, list)):
                self.states_dict[k] = tuple(to_cpu(t) for t in v)
            else:
                self.states_dict[k] = to_cpu(v)

    def prefetch(self):
        def to_device(x):
            return (
                x.to(torch.get_default_device(), non_blocking=True)
                if isinstance(x, torch.Tensor)
                else x
            )

        for k, v in self.states_dict.items():
            if v is None:
                continue
            if isinstance(v, (tuple, list)):
                self.states_dict[k] = tuple(to_device(t) for t in v)
            else:
                self.states_dict[k] = to_device(v)


class StatesDictCache(HuggingFaceCache):
    def __init__(
        self,
        layers: list[CacheLayerMixin] | None = None,
        layer_class_to_replicate: type[CacheLayerMixin] | None = None,
        **kwargs,
    ):
        if layers is None and layer_class_to_replicate is None:
            layer_class_to_replicate = StatesDictLayer
        super().__init__(
            layers=layers, layer_class_to_replicate=layer_class_to_replicate, **kwargs
        )

    def get_seq_length(self, layer_idx: int) -> int:
        """Returns the sequence length of the cache for the given layer."""
        if layer_idx >= len(self.layers):
            return 0
        return self.layers[layer_idx].get_seq_length()

    def get_states(self, layer_idx: int) -> dict[str, torch.Tensor]:
        return self.update(layer_idx=layer_idx, offset=0, states_dict=None)

    def update(
        self,
        layer_idx: int,
        offset: int = 1,
        states_dict: dict[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Updates the cache with the new `states_dict` for the layer `layer_idx`.

        Parameters:
            layer_idx (`int`):
                The index of the layer to cache the states for.

        Return:
            The updated states dict.
        """
        # In this case, the `layers` were not provided, and we must append as much as `layer_idx`
        if self.layer_class_to_replicate is not None:
            while len(self.layers) <= layer_idx:
                self.layers.append(self.layer_class_to_replicate())

        if self.offloading:
            # Wait for the stream to finish if needed, and start prefetching the next layer
            torch.cuda.default_stream(torch.get_default_device()).wait_stream(
                self.prefetch_stream
            )
            self.prefetch(layer_idx + 1, self.only_non_sliding)

        states_dict = self.layers[layer_idx].update(
            layer_idx=layer_idx, offset=offset, states_dict=states_dict, *args, **kwargs
        )

        if self.offloading:
            self.offload(layer_idx, self.only_non_sliding)

        return states_dict
