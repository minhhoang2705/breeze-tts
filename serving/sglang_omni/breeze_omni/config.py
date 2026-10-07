"""SGLang-Omni pipeline declaration for Breeze TTS 2."""

from __future__ import annotations

from typing import Any, ClassVar

from sglang_omni.config import EngineStageConfig, FactoryArgs, PipelineConfig, StageConfig

_PKG = "breeze_omni"


class BreezeTtsPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "BreezeForConditionalGeneration"
    requires_model_capabilities: ClassVar[bool] = True
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "tts_engine": EngineStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_preprocessing_executor",
            factory=FactoryArgs(max_concurrency=4),
            next="tts_engine",
        ),
        EngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory=FactoryArgs(dtype="bfloat16"),
            gpu_memory_fraction=0.75,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory_path=f"{_PKG}.stages.create_vocoder_executor",
            factory=FactoryArgs(dtype="bfloat16", device="cuda"),
            gpu=0,
            gpu_memory_fraction=0.10,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, Any]:
        return {}

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = BreezeTtsPipelineConfig
