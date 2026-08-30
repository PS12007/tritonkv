"""Model / benchmark configuration.

The kernel is benchmarked at the attention shapes of a real small dense model
rather than at round made-up numbers, because head_dim and the GQA group size
change the arithmetic intensity of the kernel and therefore the answer.

The values below are the published architecture parameters. ``load_model_config``
will verify them against the actual HuggingFace config if ``transformers`` is
installed and the model is reachable; the benchmark records whether verification
happened so the README can say so honestly.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelShape:
    name: str
    hf_id: str
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int

    @property
    def gqa_group(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    def kv_bytes_per_token(self, bits_per_element: float) -> float:
        """Bytes of KV cache per token across the whole model, at a given bitrate."""
        elems = 2 * self.num_layers * self.num_kv_heads * self.head_dim  # K and V
        return elems * bits_per_element / 8

    def as_dict(self) -> dict:
        d = asdict(self)
        d["gqa_group"] = self.gqa_group
        return d


MODELS = {
    "qwen2.5-1.5b": ModelShape(
        name="Qwen2.5-1.5B-Instruct",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
        num_layers=28,
        num_q_heads=12,
        num_kv_heads=2,
        head_dim=128,
    ),
    "llama-3.2-1b": ModelShape(
        name="Llama-3.2-1B-Instruct",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
        num_layers=16,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=64,
    ),
}

DEFAULT_MODEL = "qwen2.5-1.5b"

CONTEXT_LENGTHS = (512, 2048, 8192, 16384)

# 4-bit is the headline configuration; 2-bit is the stretch goal.
BIT_WIDTHS = (4, 2)
DEFAULT_GROUP_SIZE = 32


def load_model_config(key: str = DEFAULT_MODEL) -> tuple[ModelShape, str]:
    """Return the shape and a string saying how it was obtained.

    Never raises: if transformers is missing or offline, the hardcoded values
    are used and the caller is told so.
    """
    shape = MODELS[key]
    try:
        from transformers import AutoConfig  # type: ignore

        cfg = AutoConfig.from_pretrained(shape.hf_id)
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        actual = ModelShape(
            name=shape.name,
            hf_id=shape.hf_id,
            num_layers=cfg.num_hidden_layers,
            num_q_heads=cfg.num_attention_heads,
            num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=head_dim,
        )
        if actual == shape:
            return shape, "verified against HuggingFace config.json"
        return actual, f"loaded from HuggingFace config.json (differs from hardcoded {shape})"
    except Exception as exc:
        return shape, f"hardcoded (not verified: {type(exc).__name__})"
