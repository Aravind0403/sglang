"""Model-specific resolution declarations — see arg_groups/overrides.py."""

from typing import Any

from sglang.srt.arg_groups.overrides import (
    _is_mxfp4_pack_quantized,
    _register_for,
    logger,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform


@_register_for("KimiK3ForConditionalGeneration")
def _kimi_k3_moe_runner_overrides(server_args: Any, hf_config: Any) -> dict:
    # MoE runner default, independent of the attention-backend gate above.
    # trtllm-gen fused MoE (flashinfer_mxfp4) beats marlin on both the decode
    # (M=bs) and the target-verify (M=bs*(gamma+1)) regimes on SM100/SM103.
    # SM107 uses the same packed-MXFP4 runner; leaving auto unresolved falls
    # back to BF16 weight materialization during model loading.
    cfg = resolving_view(server_args)
    if cfg.moe_runner_backend != "auto":
        return {}
    if not (get_platform().is_sm100 and get_platform().device_sm in (100, 103, 107)):
        return {}
    if not _is_mxfp4_pack_quantized(hf_config):
        return {}
    logger.info(
        "Kimi-K3 on SM100/SM103/SM107: moe_runner_backend=flashinfer_mxfp4 "
        "(FlashInfer SiTU kernels)."
    )
    return {"moe_runner_backend": "flashinfer_mxfp4"}
