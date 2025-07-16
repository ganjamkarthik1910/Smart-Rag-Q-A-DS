"""
models package
~~~~~~~~~~~~~~
Convenience wrappers for loading the fine‑tuned Llama‑3 LoRA checkpoint
in both training and inference contexts.
"""

from pathlib import Path
from typing import Tuple
import torch, transformers, bitsandbytes as bnb
from peft import PeftModel, PeftConfig

__all__ = ["load_lora_llama", "load_tokenizer"]

def load_lora_llama(ckpt_dir: str | Path, four_bit: bool = True) -> PeftModel:
    """
    Load a LoRA‑adapted Llama‑3 model from ``ckpt_dir``.

    Parameters
    ----------
    ckpt_dir : str | Path
        Path produced by `train.py` (contains adapter weights + tokenizer).
    four_bit : bool
        If True, load the base weights in 4‑bit NF4 to fit consumer GPUs.

    Returns
    -------
    peft.PeftModel
        Ready‑to‑use model object.
    """
    ckpt_dir = Path(ckpt_dir)
    cfg = PeftConfig.from_pretrained(ckpt_dir)

    quant_args = dict(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if four_bit else {}

    base = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name_or_path, device_map="auto", **quant_args
    )
    model = PeftModel.from_pretrained(base, ckpt_dir, device_map="auto")
    model.eval()
    return model

def load_tokenizer(ckpt_dir: str | Path) -> transformers.PreTrainedTokenizer:
    """
    Return the tokenizer stored alongside the LoRA checkpoint.
    """
    return transformers.AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
