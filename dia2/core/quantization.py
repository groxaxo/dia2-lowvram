"""BitsAndBytes quantization support for low VRAM inference."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear8bitLt, Linear4bit

    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    Linear8bitLt = None
    Linear4bit = None


def check_bnb_available() -> bool:
    """Check if bitsandbytes is available for quantization."""
    return BNB_AVAILABLE


def replace_linear_with_8bit(
    model: nn.Module,
    threshold: float = 6.0,
    exclude_modules: Optional[list[str]] = None,
) -> nn.Module:
    """
    Replace nn.Linear layers with 8-bit quantized versions in-place.

    Args:
        model: The model to quantize.
        threshold: Threshold for outlier detection.
        exclude_modules: List of module name patterns to exclude.

    Returns:
        The quantized model.
    """
    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes is required for 8-bit quantization. Install with: pip install bitsandbytes")

    exclude_modules = exclude_modules or []

    for name, module in model.named_children():
        # Skip excluded modules
        if any(excl in name for excl in exclude_modules):
            continue

        if isinstance(module, nn.Linear):
            # Create 8-bit linear layer
            new_layer = Linear8bitLt(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )
            # Copy weights
            new_layer.weight = bnb.nn.Int8Params(
                module.weight.data.to(torch.float16),
                requires_grad=False,
                has_fp16_weights=False,
            )
            if module.bias is not None:
                new_layer.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, new_layer)
        else:
            # Recurse into child modules
            replace_linear_with_8bit(module, threshold, exclude_modules)

    return model


def replace_linear_with_4bit(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.bfloat16,
    quant_type: str = "nf4",
    exclude_modules: Optional[list[str]] = None,
) -> nn.Module:
    """
    Replace nn.Linear layers with 4-bit quantized versions in-place.

    Args:
        model: The model to quantize.
        compute_dtype: Data type for compute operations.
        quant_type: Quantization type ("nf4" or "fp4").
        exclude_modules: List of module name patterns to exclude.

    Returns:
        The quantized model.
    """
    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes is required for 4-bit quantization. Install with: pip install bitsandbytes")

    exclude_modules = exclude_modules or []

    for name, module in model.named_children():
        # Skip excluded modules
        if any(excl in name for excl in exclude_modules):
            continue

        if isinstance(module, nn.Linear):
            # Create 4-bit linear layer
            new_layer = Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=compute_dtype,
                quant_type=quant_type,
            )
            # Copy weights
            new_layer.weight = bnb.nn.Params4bit(
                module.weight.data.to(torch.float16),
                requires_grad=False,
                quant_type=quant_type,
            )
            if module.bias is not None:
                new_layer.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, new_layer)
        else:
            # Recurse into child modules
            replace_linear_with_4bit(module, compute_dtype, quant_type, exclude_modules)

    return model


def quantize_model(
    model: nn.Module,
    quantization: str,
    device: torch.device,
    compute_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """
    Quantize a model using the specified quantization method.

    Args:
        model: The model to quantize.
        quantization: Quantization type ("8bit", "4bit", or "none").
        device: Target device.
        compute_dtype: Data type for compute operations.

    Returns:
        The quantized model.
    """
    if quantization == "none" or not quantization:
        return model.to(device)

    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes is required for quantization. Install with: pip install bitsandbytes")

    # Exclude normalization layers and embedding layers from quantization
    exclude = ["norm", "embed", "rotary"]

    if quantization == "8bit":
        model = replace_linear_with_8bit(model, exclude_modules=exclude)
    elif quantization == "4bit":
        model = replace_linear_with_4bit(model, compute_dtype=compute_dtype, exclude_modules=exclude)
    else:
        raise ValueError(f"Unsupported quantization type: {quantization}. Use '8bit', '4bit', or 'none'.")

    return model.to(device)


__all__ = [
    "BNB_AVAILABLE",
    "check_bnb_available",
    "replace_linear_with_8bit",
    "replace_linear_with_4bit",
    "quantize_model",
]
