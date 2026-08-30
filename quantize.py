"""Group-wise low-bit quantization for KV-cache tensors.

This is the *reference* implementation: pure PyTorch, deliberately simple, and
the ground truth that the Triton kernel in ``kernels/fused_decode_attn.py`` is
checked against.

Scheme
------
Asymmetric, group-wise, along the ``head_dim`` axis (the last axis).

For a group of ``group_size`` contiguous elements ``x``::

    scale = (max(x) - min(x)) / (2**nbits - 1)      # stored fp16
    zero  = min(x)                                  # stored fp16
    code  = round((x - zero) / scale)  in [0, 2**nbits - 1]
    x_hat = code * scale + zero

Note that ``zero`` here is a *bias in value space*, not an integer zero-point in
code space. It is algebraically the same asymmetric scheme, but it avoids the
double-rounding error of an integer zero-point, and it makes the dequant a
single fused multiply-add -- which is exactly what we want inside the kernel.

``scale`` and ``zero`` are rounded to fp16 **before** the codes are computed, so
the reference dequant and the kernel's dequant see bit-identical parameters and
any mismatch between them is a real kernel bug rather than a metadata rounding
artefact.

Why quantize along ``head_dim`` and not along the token axis?
-------------------------------------------------------------
KIVI (Liu et al., 2024) shows that *keys* are better quantized per-channel
(i.e. grouping along the token axis) because key outliers are channel-aligned.
We deliberately do **not** do that: per-channel key quantization needs the group
statistics to span tokens, which forces either a re-quantization pass every time
the cache grows or a separate fp16 residual buffer for the newest tokens. Both
add machinery that has nothing to do with the thing this project is measuring
(fused dequant+attention throughput). We use per-token grouping along head_dim
for both K and V, and report the resulting accuracy honestly. See the "Known
limitations" section of the README.

Packing layout ("split-P")
--------------------------
With ``P = 8 // nbits`` codes per byte and ``DP = head_dim // P`` bytes per row,
byte ``j`` holds the codes for dims ``j, j + DP, j + 2*DP, ...`` in ascending
bit-slices::

    packed[..., j] bits [p*nbits : (p+1)*nbits]  ==  code[..., j + p*DP]

This is *not* the obvious interleaved layout (``packed[j] -> dims 2j, 2j+1``).
It is chosen because a Triton program that loads ``packed[..., 0:DP]`` gets
``P`` clean, contiguous, already-aligned sub-vectors of the head dimension for
free -- one per bit-slice -- with no shuffle or de-interleave step. The
dequantized sub-vectors line up with matching slices of ``q``, so each one
feeds straight into a ``tl.dot``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

SUPPORTED_BITS = (2, 4)

# Metadata (scale + zero) is stored in fp16. 2 params x 16 bits per group.
_META_BITS_PER_GROUP = 2 * 16


def _check_bits(nbits: int) -> None:
    if nbits not in SUPPORTED_BITS:
        raise ValueError(f"nbits must be one of {SUPPORTED_BITS}, got {nbits}")


@dataclass
class QuantizedTensor:
    """A group-wise quantized tensor plus everything needed to invert it.

    Attributes
    ----------
    packed:
        ``uint8``, shape ``(*batch_dims, head_dim // P)``.
    scale, zero:
        ``float16``, shape ``(*batch_dims, head_dim // group_size)``.
    """

    packed: torch.Tensor
    scale: torch.Tensor
    zero: torch.Tensor
    nbits: int
    group_size: int
    head_dim: int

    @property
    def codes_per_byte(self) -> int:
        return 8 // self.nbits

    @property
    def n_groups(self) -> int:
        return self.head_dim // self.group_size

    @property
    def batch_shape(self) -> torch.Size:
        return self.packed.shape[:-1]

    @property
    def shape(self) -> torch.Size:
        return torch.Size((*self.batch_shape, self.head_dim))

    @property
    def device(self) -> torch.device:
        return self.packed.device

    def nbytes(self) -> int:
        """Actual bytes of storage, including metadata."""
        return (
            self.packed.numel() * self.packed.element_size()
            + self.scale.numel() * self.scale.element_size()
            + self.zero.numel() * self.zero.element_size()
        )

    def effective_bits_per_element(self) -> float:
        """Real bits/element *including* the fp16 scale and zero.

        This is the number that should be quoted, not ``nbits``. 4-bit with
        ``group_size=32`` is really 5.0 bits/element, not 4.
        """
        return self.nbits + _META_BITS_PER_GROUP / self.group_size

    def to(self, *args, **kwargs) -> "QuantizedTensor":
        return QuantizedTensor(
            packed=self.packed.to(*args, **kwargs),
            scale=self.scale.to(*args, **kwargs),
            zero=self.zero.to(*args, **kwargs),
            nbits=self.nbits,
            group_size=self.group_size,
            head_dim=self.head_dim,
        )

    def contiguous(self) -> "QuantizedTensor":
        return QuantizedTensor(
            packed=self.packed.contiguous(),
            scale=self.scale.contiguous(),
            zero=self.zero.contiguous(),
            nbits=self.nbits,
            group_size=self.group_size,
            head_dim=self.head_dim,
        )


def pack_codes(codes: torch.Tensor, nbits: int) -> torch.Tensor:
    """Pack ``uint8`` codes into the split-P byte layout described above.

    ``codes``: ``(..., D)`` uint8, values in ``[0, 2**nbits)``.
    Returns ``(..., D // P)`` uint8.
    """
    _check_bits(nbits)
    if codes.dtype != torch.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype}")
    P = 8 // nbits
    D = codes.shape[-1]
    if D % P != 0:
        raise ValueError(f"head_dim {D} must be divisible by {P} for {nbits}-bit packing")

    # (..., P, DP): chunk p is codes[..., p*DP : (p+1)*DP] -- exactly the
    # split-P layout, for free, because the view is over contiguous memory.
    chunks = codes.reshape(*codes.shape[:-1], P, D // P)
    packed = torch.zeros(chunks.shape[:-2] + chunks.shape[-1:], dtype=torch.uint8, device=codes.device)
    for p in range(P):
        packed |= chunks[..., p, :] << (p * nbits)
    return packed


def unpack_codes(packed: torch.Tensor, nbits: int, head_dim: int) -> torch.Tensor:
    """Inverse of :func:`pack_codes`. Returns ``(..., head_dim)`` uint8."""
    _check_bits(nbits)
    P = 8 // nbits
    DP = head_dim // P
    if packed.shape[-1] != DP:
        raise ValueError(f"packed last dim {packed.shape[-1]} != head_dim//P = {DP}")
    mask = (1 << nbits) - 1
    shifts = torch.arange(P, device=packed.device, dtype=torch.uint8) * nbits
    # (..., P, DP) -> (..., D)
    out = (packed.unsqueeze(-2) >> shifts.view(P, 1)) & mask
    return out.reshape(*packed.shape[:-1], head_dim).contiguous()


def quantize_groupwise(x: torch.Tensor, nbits: int, group_size: int) -> QuantizedTensor:
    """Group-wise asymmetric quantization along the last axis.

    Parameters
    ----------
    x:
        ``(..., head_dim)`` float tensor.
    nbits:
        2 or 4.
    group_size:
        Must divide ``head_dim``.
    """
    _check_bits(nbits)
    D = x.shape[-1]
    if group_size <= 0 or D % group_size != 0:
        raise ValueError(f"group_size {group_size} must divide head_dim {D}")
    P = 8 // nbits
    if D % P != 0:
        raise ValueError(f"head_dim {D} must be divisible by {P} for {nbits}-bit packing")

    qmax = (1 << nbits) - 1
    n_groups = D // group_size

    xg = x.reshape(*x.shape[:-1], n_groups, group_size).float()
    gmin = xg.amin(dim=-1)
    gmax = xg.amax(dim=-1)

    scale = (gmax - gmin) / qmax
    # Degenerate (constant) group: scale 0 would divide by zero. Setting the
    # scale to 1 makes every code 0 and dequant returns exactly `zero == gmin`,
    # which is the exact answer for a constant group.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    # Round metadata to fp16 *first* so reference and kernel agree bit-exactly.
    scale16 = scale.half()
    zero16 = gmin.half()
    scale_f = scale16.float()
    zero_f = zero16.float()

    codes = torch.round((xg - zero_f.unsqueeze(-1)) / scale_f.unsqueeze(-1))
    codes = codes.clamp_(0, qmax).to(torch.uint8)
    codes = codes.reshape(*x.shape[:-1], D)

    return QuantizedTensor(
        packed=pack_codes(codes, nbits),
        scale=scale16,
        zero=zero16,
        nbits=nbits,
        group_size=group_size,
        head_dim=D,
    )


def dequantize_groupwise(qt: QuantizedTensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Inverse of :func:`quantize_groupwise`. Returns ``(..., head_dim)``."""
    codes = unpack_codes(qt.packed, qt.nbits, qt.head_dim)
    cg = codes.reshape(*codes.shape[:-1], qt.n_groups, qt.group_size).to(torch.float32)
    scale = qt.scale.float().unsqueeze(-1)
    zero = qt.zero.float().unsqueeze(-1)
    out = cg * scale + zero
    return out.reshape(*codes.shape[:-1], qt.head_dim).to(dtype)


def quantize_kv(
    k: torch.Tensor, v: torch.Tensor, nbits: int, group_size: int
) -> tuple[QuantizedTensor, QuantizedTensor]:
    """Quantize a K and a V cache independently (they get their own stats)."""
    return (
        quantize_groupwise(k, nbits, group_size).contiguous(),
        quantize_groupwise(v, nbits, group_size).contiguous(),
    )


def roundtrip_error(x: torch.Tensor, nbits: int, group_size: int) -> dict[str, float]:
    """Quantize/dequantize ``x`` and report the reconstruction error."""
    qt = quantize_groupwise(x, nbits, group_size)
    xh = dequantize_groupwise(qt, dtype=torch.float32)
    xf = x.float()
    err = (xh - xf)
    denom = xf.norm().clamp_min(1e-12)
    return {
        "max_abs_err": err.abs().max().item(),
        "rel_l2": (err.norm() / denom).item(),
        "cosine": torch.nn.functional.cosine_similarity(
            xh.flatten().unsqueeze(0), xf.flatten().unsqueeze(0)
        ).item(),
        "effective_bits": qt.effective_bits_per_element(),
        "bytes": qt.nbytes(),
        "fp16_bytes": x.numel() * 2,
    }
