"""Register Breeze with Transformers, SGLang-Omni, and SGLang."""
from __future__ import annotations

import importlib
import os
from typing import Any


_REGISTERED = False


def _load_symbol(path: str) -> Any:
    module_name, separator, symbol_name = path.partition(":")
    if not separator or not module_name or not symbol_name:
        raise ValueError(f"Expected module:Symbol import path, got {path!r}")
    return getattr(importlib.import_module(module_name), symbol_name)


def register() -> None:
    """Install model-family registrations; safe to call repeatedly in each process."""
    global _REGISTERED
    if _REGISTERED:
        return

    # The optional overrides also let the boot hook be verified with isolated
    # throwaway classes without adding test-only files to this package.
    model_type = os.getenv("BREEZE_OMNI_MODEL_TYPE", "breeze")
    architecture = os.getenv(
        "BREEZE_OMNI_ARCHITECTURE", "BreezeForConditionalGeneration"
    )
    hf_config_path = os.getenv(
        "BREEZE_OMNI_HF_CONFIG", "breeze_omni.hf_config:BreezeHFConfig"
    )
    pipeline_path = os.getenv(
        "BREEZE_OMNI_PIPELINE_CONFIG",
        "breeze_omni.config:BreezeTtsPipelineConfig",
    )
    model_path = os.getenv(
        "BREEZE_OMNI_MODEL", "breeze_omni.model:BreezeTTSModel"
    )

    from transformers import AutoConfig

    hf_config_cls = _load_symbol(hf_config_path)
    try:
        AutoConfig.register(model_type, hf_config_cls)
    except ValueError:
        # AutoConfig.register rejects even an idempotent duplicate.
        current = AutoConfig.for_model(model_type).__class__
        if current is not hf_config_cls:
            raise

    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    pipeline_cls = _load_symbol(pipeline_path)
    PIPELINE_CONFIG_REGISTRY.configs[architecture] = pipeline_cls

    from sglang.srt.models.registry import ModelRegistry

    model_cls = _load_symbol(model_path)
    ModelRegistry.models[architecture] = model_cls

    # The Omni runner explicitly re-registers its built-in classes at startup.
    # Preserve the out-of-tree entry if a future Omni release adds the same
    # architecture to that list.
    from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner

    current = SGLModelRunner._register_omni_model
    if not getattr(current, "_breeze_omni_wrapped", False):
        def register_then_restore(runner: Any) -> None:
            current(runner)
            ModelRegistry.models[architecture] = model_cls

        register_then_restore._breeze_omni_wrapped = True  # type: ignore[attr-defined]
        SGLModelRunner._register_omni_model = register_then_restore

    _REGISTERED = True
