from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        full_determinism=getattr(args, "full_determinism", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    # Prefer explicit pad_token_id if provided; fallback to legacy pad_token_string for backward compatibility
    if strategy is not None and hasattr(strategy, "args"):
        pad_id = getattr(strategy.args, "pad_token_id", None)
        if pad_id is not None:
            pad_id = int(pad_id)
            pad_tok = tokenizer.convert_ids_to_tokens(pad_id)
            # Require a valid string-like token mapping
            if pad_tok is None or not isinstance(pad_tok, str) or len(pad_tok) == 0:
                raise ValueError(
                    f"pad_token_id {pad_id} does not map to a valid string token via tokenizer.convert_ids_to_tokens"
                )
            tokenizer.pad_token_id = pad_id
            tokenizer.pad_token = pad_tok
            if model is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
        elif getattr(strategy.args, "pad_token_string", None):
            # Legacy behavior
            pad_token_str = getattr(strategy.args, "pad_token_string")
            token_ids = tokenizer.encode(pad_token_str, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"pad_token_string '{pad_token_str}' must map to exactly one token id, got ids={token_ids}"
                )
            tokenizer.pad_token = pad_token_str
            tokenizer.pad_token_id = token_ids[0]
            if model is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
    # Otherwise, ensure pad token is set (fallback to eos)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if model is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)


def remove_pad_token(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Remove the pad token. Return tensors and not lists.

    Args:
        input_ids shape: [bs, seq_length]
        attention_mask shape: [bs, seq_length]
    Returns:
        no_padding_batch(List[Tensor[int]]): contains the rmpad token ids per query.
    """
    no_padding_batch = []
    for ids, mask in zip(input_ids, attention_mask):
        # Fix for both left and right padding
        no_padding_batch.append((ids[mask.bool()]))
    return no_padding_batch
