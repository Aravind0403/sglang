"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    _register_for,
    logger,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform


@_register_for("MuseGlimmerForConditionalGeneration", "MuseGlimmerForCausalLM")
def _muse_glimmer_fp4_gemm_runner_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if get_platform().is_sm120 and cfg.fp4_gemm_runner_backend == "auto":
        logger.info("Use marlin as FP4 GEMM runner backend on SM120 for Muse Glimmer")
        return {"fp4_gemm_runner_backend": "marlin"}
    return {}
