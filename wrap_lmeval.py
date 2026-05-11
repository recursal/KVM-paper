from utils.env import setup_env

if __name__ == "__main__":
    setup_env()

import copy

import torch
import transformers.generation
from transformers import AutoModelForCausalLM, AttentionInterface, GenerationMixin
from transformers.generation.configuration_utils import GenerationConfig

from lm_eval.models.utils_hf import MultiTokenEOSCriteria, stop_sequences_criteria


def _right_pad_sequence_tensor(
    tensor: torch.Tensor, target_length: int, pad_token_id: int
) -> torch.Tensor:
    if tensor.shape[1] >= target_length:
        return tensor
    padding = torch.full(
        (tensor.shape[0], target_length - tensor.shape[1]),
        fill_value=pad_token_id,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=1)


def _left_pad_sequence_tensor(
    tensor: torch.Tensor, pad_length: int, pad_token_id: int
) -> torch.Tensor:
    if pad_length <= 0:
        return tensor
    padding = torch.full(
        (tensor.shape[0], pad_length),
        fill_value=pad_token_id,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([padding, tensor], dim=1)


def _rebuild_stopping_criteria(
    stopping_criteria, initial_decoder_input_length, batch_size: int
):
    if stopping_criteria is None:
        return None

    return [
        MultiTokenEOSCriteria(
            criterion.sequence,
            criterion.tokenizer,
            initial_decoder_input_length,
            batch_size,
        )
        if isinstance(criterion, MultiTokenEOSCriteria)
        else criterion
        for criterion in stopping_criteria
    ]


prefill_chunk_size = 256


class GenerationMixinReplacement(GenerationMixin):
    @torch.no_grad()
    def generate(
        self,
        inputs: torch.Tensor | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ):  # -> GenerateOutput | torch.LongTensor:
        ## remove idiotic generation config settings from qwen
        # kwargs['max_new_tokens'] = None
        # set up batched prefill
        # kwargs['prefill_chunk_size'] = prefill_chunk_size
        # Preserve caller-provided generation length settings. Left-pad trimming
        # below adjusts absolute-length controls per subgroup when needed.

        input_ids = kwargs.get("input_ids")
        if input_ids is None and isinstance(inputs, torch.Tensor):
            input_ids = inputs

        attention_mask = kwargs.get("attention_mask")
        if (
            input_ids is not None
            and isinstance(input_ids, torch.Tensor)
            and input_ids.ndim == 2
            and attention_mask is not None
            and isinstance(attention_mask, torch.Tensor)
            and attention_mask.ndim == 2
            and input_ids.shape == attention_mask.shape
            and input_ids.shape[0] > 1
        ):
            prompt_lengths = attention_mask.sum(dim=-1)
            unique_lengths = torch.unique(prompt_lengths)
            if unique_lengths.numel() > 1:
                input_ids = kwargs.pop("input_ids")
                outputs_by_index = {}
                for prompt_length in unique_lengths.tolist():
                    group_mask = prompt_lengths == prompt_length
                    batch_indices = torch.nonzero(group_mask, as_tuple=False).view(-1)
                    group_input_ids = input_ids.index_select(0, batch_indices)

                    def _select_batch_item(key, value, batch_indices: torch.Tensor):
                        if (
                            isinstance(value, torch.Tensor)
                            and value.ndim > 0
                            and value.shape[0] == batch_indices.shape[0]
                        ):
                            return value
                        if (
                            isinstance(value, torch.Tensor)
                            and value.ndim > 0
                            and value.shape[0] >= int(batch_indices.max().item()) + 1
                        ):
                            return value.index_select(0, batch_indices)
                        return copy.deepcopy(value)

                    group_kwargs = {
                        key: _select_batch_item(key, value, batch_indices)
                        for key, value in kwargs.items()
                    }
                    group_generation_config = copy.deepcopy(generation_config)

                    shared_left_pad = group_input_ids.shape[1] - int(prompt_length)
                    if shared_left_pad > 0:
                        pad_token_id = group_input_ids[0, 0]
                        group_input_ids = group_input_ids[:, shared_left_pad:]
                        trimmed_attention_mask = group_kwargs.get("attention_mask")
                        if (
                            isinstance(trimmed_attention_mask, torch.Tensor)
                            and trimmed_attention_mask.ndim == 2
                        ):
                            group_kwargs["attention_mask"] = trimmed_attention_mask[
                                :, shared_left_pad:
                            ]
                        if "position_ids" in group_kwargs:
                            group_kwargs.pop("position_ids")
                        if "cache_position" in group_kwargs:
                            group_kwargs.pop("cache_position")
                        if group_generation_config is not None:
                            if (
                                getattr(group_generation_config, "max_length", None)
                                is not None
                            ):
                                group_generation_config.max_length = max(
                                    int(group_generation_config.max_length)
                                    - shared_left_pad,
                                    group_input_ids.shape[1],
                                )
                            if (
                                getattr(group_generation_config, "min_length", None)
                                is not None
                            ):
                                group_generation_config.min_length = max(
                                    int(group_generation_config.min_length)
                                    - shared_left_pad,
                                    0,
                                )
                        for key in ("max_length", "min_length"):
                            if key in group_kwargs and group_kwargs[key] is not None:
                                group_kwargs[key] = max(
                                    int(group_kwargs[key]) - shared_left_pad, 0
                                )

                    if "stopping_criteria" in group_kwargs:
                        group_kwargs["stopping_criteria"] = _rebuild_stopping_criteria(
                            group_kwargs["stopping_criteria"],
                            initial_decoder_input_length=group_input_ids.shape[1],
                            batch_size=batch_indices.numel(),
                        )

                    assert (
                        group_input_ids is not None and group_input_ids.shape[0] > 0
                    ), (
                        f"Invalid group_input_ids shape after processing: {group_input_ids.shape}"
                    )
                    group_output = super().generate(
                        inputs=None,
                        input_ids=group_input_ids,
                        generation_config=group_generation_config,
                        **group_kwargs,
                    )

                    if isinstance(group_output, torch.Tensor):
                        group_output = _left_pad_sequence_tensor(
                            group_output, shared_left_pad, pad_token_id
                        )
                        for local_idx, batch_idx in enumerate(batch_indices.tolist()):
                            outputs_by_index[batch_idx] = group_output[
                                local_idx : local_idx + 1
                            ]
                    elif hasattr(group_output, "sequences"):
                        group_sequences = _left_pad_sequence_tensor(
                            group_output.sequences, shared_left_pad, pad_token_id
                        )
                        for local_idx, batch_idx in enumerate(batch_indices.tolist()):
                            outputs_by_index[batch_idx] = group_sequences[
                                local_idx : local_idx + 1
                            ]
                    else:
                        raise TypeError(
                            f"Unsupported generate output type for regrouping: {type(group_output)!r}"
                        )

                # reorder and generate output
                batch_size = input_ids.shape[0]
                first_output = next(iter(outputs_by_index.values()))
                if isinstance(first_output, torch.Tensor):
                    target_length = max(
                        outputs_by_index[i].shape[1] for i in range(batch_size)
                    )
                    return torch.cat(
                        [
                            _right_pad_sequence_tensor(
                                outputs_by_index[i], target_length, pad_token_id
                            )
                            for i in range(batch_size)
                        ],
                        dim=0,
                    )

                raise TypeError(
                    f"Unsupported generate output type for regrouping: {type(first_output)!r}"
                )

        return super().generate(inputs, generation_config=generation_config, **kwargs)

    ### BEGIN NEW CODE ADDED TO HF CODE ###
    def model_forward(self, **kwargs):
        return self(**kwargs)

    ### END NEW CODE ADDED TO HF CODE ###

    # code from 5.3.0
    # TODO: v5.1: make public once API stabilized
    def _prefill(
        self,
        input_ids: torch.LongTensor,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        is_first_iteration: bool = True,
    ):
        # NOTE - tracking attention sinks
        attention_mask = model_kwargs["attention_mask"]
        B, S = attention_mask.shape
        sink_indices = (S - attention_mask.sum(dim=-1)).view(
            B
        )  # sink offset per batch idx
        print("sinks", sink_indices)
        ### END NEW CODE ADDED TO HF CODE ###

        """
        Perform the prefill stage of generation.

        Note that usually, the prefill stage is always the first iteration of a new input batch, and thus multimodal inputs etc
        should be treated as if it's the first iteration. However, for assisted decoding, assistants call `generate`
        several time in a row for a same batch of inputs, so we need to pass `is_first_iteration` here for such cases.
        """
        # When restarting from previous cache, the `input_ids` are either the FULL sequence, including previous inputs,
        # or only the new tokens but in this case the attention_mask still contains the FULL sequence (because otherwise we may
        # lose some early padding tokens information). So slice inputs according to that if needed
        # When restarting from `inputs_embeds`, it's always the FULL sequence, and we always need to slice

        # cache = model_kwargs.get("past_key_values")
        # if cache is None or not isinstance(cache, StatesDictCache):
        #     print(f"REPLACING {cache} with StatesDictCache()")
        #     model_kwargs['past_key_values'] = cache = StatesDictCache()

        next_sequence_length = None
        past_length = 0
        inputs_embeds = model_kwargs.get("inputs_embeds")
        use_inputs_embeds = False
        if (
            not self.config.is_encoder_decoder
            and inputs_embeds is not None
            and is_first_iteration
        ):
            use_inputs_embeds = True
        if model_kwargs.get("past_key_values") is not None:
            attention_mask_key = (
                "decoder_attention_mask"
                if self.config.is_encoder_decoder
                else "attention_mask"
            )
            current_input_length = (
                inputs_embeds.shape[1] if use_inputs_embeds else input_ids.shape[1]
            )
            attention_mask = model_kwargs.get(attention_mask_key)
            cache_position = model_kwargs.get("cache_position")
            position_ids = model_kwargs.get("position_ids")
            if cache_position is not None and cache_position.numel() > 0:
                past_length = int(cache_position.reshape(-1)[0].item())
            elif position_ids is not None and position_ids.numel() > 0:
                past_length = int(position_ids.reshape(-1)[0].item())
            elif (
                attention_mask is not None
                and attention_mask.shape[-1] > current_input_length
            ):
                past_length = int(attention_mask.shape[-1] - current_input_length)
            elif is_first_iteration:
                past_length = 0
            else:
                raise ValueError(
                    "Cannot infer cached token count from past_key_values alone. "
                    "Pass cache_position or position_ids when using a layer-indexed cache."
                )
            # Always directly slice the inputs_embeds if present, as `prepare_inputs_for_generation` never need them full and `_get_initial_cache_position`
            # rely on its size explicitly. For input_ids, we need to use `next_sequence_length` to slice later instead of explicit slicing,
            # as some model need them full for correct input preparation inside `prepare_inputs_for_generation` (i.e. audio models)
            if use_inputs_embeds:
                model_kwargs["inputs_embeds"] = inputs_embeds[:, past_length:, :]
            else:
                # In this case we need to slice - if it's smaller than the mask, only the new inputs were passed -> no need to do anything
                if (
                    attention_mask is not None
                    and input_ids.shape[1] == attention_mask.shape[1]
                ):
                    # inputs will be sliced as `input_ids[:, -next_sequence_length :]` in `prepare_inputs_for_generation`
                    next_sequence_length = input_ids.shape[1] - past_length

        # Usual prefill
        if generation_config.prefill_chunk_size is None:
            # The cache is already taken into account in `_get_initial_cache_position`, so the length is only the new tokens if we slice
            effective_input_length = (
                next_sequence_length
                if next_sequence_length is not None
                else input_ids.shape[1]
            )
            model_kwargs.setdefault(
                "cache_position",
                torch.arange(
                    past_length,
                    past_length + effective_input_length,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
            )
            model_kwargs = self._get_initial_cache_position(
                effective_input_length, input_ids.device, model_kwargs
            )
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                next_sequence_length=next_sequence_length,
                is_first_iteration=is_first_iteration,
                **model_kwargs,
            )
            return self(**model_inputs, return_dict=True)

        # Chunked prefill (for very large contexts)
        else:
            ### BEGIN NEW CODE ADDED TO HF CODE ###
            # NOTE - tracking attention sinks
            attention_mask = model_kwargs["attention_mask"]
            B, S = attention_mask.shape
            sink_indices = (S - attention_mask.sum(dim=-1)).view(
                B
            )  # sink offset per batch idx
            print("sinks", sink_indices)
            # kv_idx = torch.arange(S, device=attention_mask.device)[None, :]
            # sink_mask = kv_idx == sink_indices.view(B,1)
            model_kwargs["sink_indices"] = sink_indices
            ### END NEW CODE ADDED TO HF CODE ###

            # Even if we are not compiling the forward, flex is always compiled when used. With chunked prefill, we may
            # end up needing just a bit more graphs than the default (which is 8). Doing this avoids very cryptic warnings
            torch._dynamo.config.cache_size_limit = 64

            chunk_size = generation_config.prefill_chunk_size
            input_chunks = torch.split(input_ids, chunk_size, dim=-1)

            if "past_key_values" not in model_kwargs:
                raise ValueError("Cannot use prefill chunking without a cache")

            ### BEGIN CHANGE TO HF CODE ###
            # model_forward = (
            #     self.get_compiled_call(generation_config.compile_config)
            #     if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            #     else self.__call__
            # )
            ### END CHANGE TO HF CODE ###

            attention_mask = model_kwargs.pop("attention_mask", None)
            position_ids = model_kwargs.pop("position_ids", None)
            past_length = 0
            for input_chunk in input_chunks:
                current_length = past_length + input_chunk.shape[-1]
                if attention_mask is not None:
                    model_kwargs["attention_mask"] = attention_mask[:, :current_length]
                if position_ids is not None:
                    model_kwargs["position_ids"] = position_ids[
                        :, past_length:current_length
                    ]
                model_kwargs["cache_position"] = torch.arange(
                    past_length,
                    current_length,
                    dtype=torch.long,
                    device=input_chunk.device,
                )
                model_inputs = self.prepare_inputs_for_generation(
                    input_chunk, **model_kwargs
                )

                ### BEGIN CHANGE TO HF CODE ###
                outputs = self.model_forward(**model_inputs, return_dict=True)
                ### END CHANGE TO HF CODE ###

                model_kwargs["past_key_values"] = outputs.past_key_values
                past_length = current_length

            # Recreate the kwargs based on the full length
            model_kwargs["attention_mask"] = attention_mask
            model_kwargs["cache_position"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
            model_kwargs["position_ids"] = position_ids

            # Latest outputs contain next token logits
            return outputs


# replace the original GenerationMixin with our replacement
transformers.GenerationMixin = GenerationMixinReplacement
transformers.generation.GenerationMixin = GenerationMixinReplacement
# import the model after replacement so that the model class is registered in the AutoModelForCausalLM registry and it uses the GenerationMixinReplacement
from model.rwkv7_backbone import RWKV7BackboneForCausalLM, RWKV7BackboneModel

# NOTE - instead of knowing which model we need up front at this point, we could set a base path and have a modeling python file that imports the model like the following:
# NOTE - this would also allow us to call programs from commandline by just setting HF_WRAPPER_BASE_PATH instead of needing a wrapper, but this wrapper does force chunked prefill
# import os, sys
# hf_wrapper_base_path = os.environ.get('HF_WRAPPER_BASE_PATH')
# if hf_wrapper_base_path is not None:
#     sys.path.insert(0, hf_wrapper_base_path)
# from model.rwkv7_backbone import RWKV7BackboneForCausalLM, RWKV7BackboneModel, RWKV7BackboneConfig
# RWKV7BackboneForCausalLM = RWKV7BackboneForCausalLM
# RWKV7BackboneModel = RWKV7BackboneModel
# RWKV7BackboneConfig = RWKV7BackboneConfig

# NOTE - if so, use this code too:
# import os
# _here = os.path.dirname(os.path.abspath(__file__))
# os.environ['HF_WRAPPER_BASE_PATH'] = _here


from lm_eval.__main__ import cli_evaluate  # if we want to use lm-eval-harness

if __name__ == "__main__":
    cli_evaluate()
