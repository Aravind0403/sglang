"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    _register_for,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform


@_register_for("MiniCPMV4_6ForConditionalGeneration")
def _minicpm_v4_6_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    if get_platform().is_sm100 and cfg.attention_backend is None:
        return {"attention_backend": "triton"}
    return {}
