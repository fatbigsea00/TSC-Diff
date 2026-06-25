"""
Map-Guided Cross-Attention (MapCA) for ControlNet-Seg Training.

The module modulates frozen SD UNet cross-attention outputs with a continuous
modulation map G_r(t) = 1 + g(t) * (alpha_r * M^tar + beta_r * M^shd) derived
from the TRM colormap. It introduces no trainable network parameters; only a
small set of scalar hyperparameters (per-resolution boosts and a sigmoid time
gate). Inspired by the mask-aware cross-attention of AeroGen, but acts on the
attention output instead of attention logits.

Naming note: called "Map-Guided" rather than "Mask-Guided" because the
modulation target is the continuous map G_r(t) derived from the binary masks,
not the binary masks themselves.

V-CA:  uniform boost across all resolutions and timesteps.
V-CA1: per-resolution boost + sigmoid timestep gating.
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F


# ── Shared store ────────────────────────────────────────────────────────
# Keys: int (spatial resolution) → Tensor [B, H*W, 1]
# Special key "gate" → float scalar (timestep gate value, 0~1)

_MAP_CA_STORE: Dict = {}

# Per-resolution default boost configs
# (obj_boost, shadow_boost) for each spatial resolution
BOOST_UNIFORM = {64: (0.5, 0.3), 32: (0.5, 0.3), 16: (0.5, 0.3), 8: (0.5, 0.3)}
BOOST_LAYERED = {64: (0.1, 0.05), 32: (0.3, 0.15), 16: (0.5, 0.3), 8: (0.8, 0.5)}
BOOST_MILD    = {64: (0.35, 0.2), 32: (0.5, 0.3), 16: (0.5, 0.3), 8: (0.5, 0.3)}


def set_map_ca_data(
    colormap: torch.Tensor,
    obj_boost: Union[float, Dict[int, float]] = 0.5,
    shadow_boost: Union[float, Dict[int, float]] = 0.3,
    timestep: Optional[torch.Tensor] = None,
    gate_mid: float = 400.0,
    gate_temp: float = 100.0,
) -> None:
    """Compute modulation maps and store for processors to read.

    Args:
        colormap: [B, 3, H, W] float tensor in [0, 1].
        obj_boost:  float (uniform) or dict {res: boost} (per-resolution).
        shadow_boost: same as obj_boost.
        timestep: [B] int tensor, current diffusion timestep. None = no gating.
        gate_mid: timestep center for sigmoid gate.
        gate_temp: temperature for sigmoid gate.
    """
    global _MAP_CA_STORE

    obj_mask, shadow_mask = extract_region_masks(colormap)

    # Timestep gate: sigmoid((t - mid) / temp), high t → ~1, low t → ~0
    if timestep is not None:
        t_float = timestep.float().mean()
        gate = torch.sigmoid((t_float - gate_mid) / gate_temp).item()
    else:
        gate = 1.0

    store: Dict = {"gate": gate}
    for res in (64, 32, 16, 8):
        if isinstance(obj_boost, dict):
            ob = obj_boost.get(res, 0.5)
        else:
            ob = obj_boost
        if isinstance(shadow_boost, dict):
            sb = shadow_boost.get(res, 0.3)
        else:
            sb = shadow_boost

        obj_r = F.interpolate(obj_mask, size=(res, res), mode="bilinear", align_corners=False)
        shd_r = F.interpolate(shadow_mask, size=(res, res), mode="bilinear", align_corners=False)

        # gate scales the boost: at low timesteps, boost → 0
        mod = 1.0 + gate * (ob * obj_r + sb * shd_r)
        store[res] = mod.reshape(mod.shape[0], res * res, 1)

    _MAP_CA_STORE = store


def clear_map_ca_data() -> None:
    global _MAP_CA_STORE
    _MAP_CA_STORE = {}


# ── Region mask extraction ──────────────────────────────────────────────

def extract_region_masks(
    colormap: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract object and shadow binary masks from a colormap tensor.

    Args:
        colormap: [B, C, H, W] float tensor in [0, 1].
                  C=3 for TRM colormap; C=1 for binary mask.

    Returns:
        object_mask:  [B, 1, H, W] float, 1 at object pixels.
        shadow_mask:  [B, 1, H, W] float, 1 at shadow pixels.
    """
    if colormap.shape[1] == 1:
        object_mask = (colormap > 0.5).float()
        shadow_mask = torch.zeros_like(object_mask)
        return object_mask, shadow_mask

    max_ch = colormap.max(dim=1, keepdim=True)[0]
    min_ch = colormap.min(dim=1, keepdim=True)[0]

    object_mask = ((max_ch > 0.8) & (min_ch < 0.2)).float()

    mean_ch = colormap.mean(dim=1, keepdim=True)
    ch_var = colormap.var(dim=1, keepdim=True)
    shadow_mask = (
        (mean_ch > 0.45) & (mean_ch < 0.55) & (ch_var < 0.003)
    ).float()

    return object_mask, shadow_mask


# ── Custom attention processor ──────────────────────────────────────────

class MapGuidedAttnProcessor:
    """Drop-in replacement for AttnProcessor2_0 on cross-attention layers.

    Reads modulation maps from the module-level _MAP_CA_STORE dict.
    When no data is stored, behaves identically to AttnProcessor2_0.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask,
            dropout_p=0.0, is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # ── Map-guided output modulation ────────────────────────────
        if _MAP_CA_STORE:
            seq_len = hidden_states.shape[1]
            H = int(math.sqrt(seq_len))
            modulation = _MAP_CA_STORE.get(H)
            if modulation is not None:
                B_mod = modulation.shape[0]
                B_actual = hidden_states.shape[0]
                if B_actual > B_mod:
                    ones = torch.ones_like(modulation)
                    modulation = torch.cat([ones, modulation], dim=0)
                hidden_states = hidden_states * modulation.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )

        # ── Standard output projection ──────────────────────────────
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# ── UNet injection helpers ──────────────────────────────────────────────

def apply_map_guided_attention(unet) -> None:
    """Replace all cross-attention (attn2) processors with MapGuidedAttnProcessor."""
    attn_procs = {}
    count = 0
    for name, proc in unet.attn_processors.items():
        if "attn2" in name:
            attn_procs[name] = MapGuidedAttnProcessor()
            count += 1
        else:
            attn_procs[name] = proc
    unet.set_attn_processor(attn_procs)
    print(f"  MapGuidedCrossAttention (MapCA): replaced {count} cross-attention processors")


_UNET_HOOK_HANDLE = None


def install_timestep_hook(unet) -> None:
    """Install a forward pre-hook on UNet to capture the current timestep.

    During inference the pipeline calls unet(sample, timestep, ...).
    This hook reads the timestep arg and updates _MAP_CA_STORE['gate']
    so processors can apply timestep-aware gating.
    """
    global _UNET_HOOK_HANDLE

    def _hook(module, args, kwargs):
        # UNet forward signature: (sample, timestep, encoder_hidden_states, ...)
        # timestep is the 2nd positional arg
        if len(args) >= 2:
            t = args[1]
        elif "timestep" in kwargs:
            t = kwargs["timestep"]
        else:
            return

        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t])

        # Retrieve gate params stored on the module by install call
        gate_mid = getattr(module, "_map_ca_gate_mid", 400.0)
        gate_temp = getattr(module, "_map_ca_gate_temp", 100.0)

        t_float = t.float().mean()
        gate = torch.sigmoid((t_float - gate_mid) / gate_temp).item()

        # Update existing store's modulation maps with new gate
        if _MAP_CA_STORE and "base_obj" in _MAP_CA_STORE:
            for res in (64, 32, 16, 8):
                base_key = f"base_{res}"
                if base_key in _MAP_CA_STORE:
                    base = _MAP_CA_STORE[base_key]  # (obj_r, shd_r, ob, sb)
                    obj_r, shd_r, ob, sb = base
                    mod = 1.0 + gate * (ob * obj_r + sb * shd_r)
                    _MAP_CA_STORE[res] = mod.reshape(mod.shape[0], res * res, 1)

        _MAP_CA_STORE["gate"] = gate

    if _UNET_HOOK_HANDLE is not None:
        _UNET_HOOK_HANDLE.remove()
    _UNET_HOOK_HANDLE = unet.register_forward_pre_hook(_hook, with_kwargs=True)


def set_map_ca_data_with_bases(
    colormap: torch.Tensor,
    obj_boost_dict: Dict[int, float],
    shadow_boost_dict: Dict[int, float],
) -> None:
    """Store mask data with base masks for timestep hook to recompute modulation.

    Used for inference where timestep changes each denoising step.
    """
    global _MAP_CA_STORE

    obj_mask, shadow_mask = extract_region_masks(colormap)

    store: Dict = {"gate": 1.0, "base_obj": True}
    for res in (64, 32, 16, 8):
        ob = obj_boost_dict.get(res, 0.5)
        sb = shadow_boost_dict.get(res, 0.3)
        obj_r = F.interpolate(obj_mask, size=(res, res), mode="bilinear", align_corners=False)
        shd_r = F.interpolate(shadow_mask, size=(res, res), mode="bilinear", align_corners=False)
        store[f"base_{res}"] = (obj_r, shd_r, ob, sb)
        mod = 1.0 + ob * obj_r + sb * shd_r
        store[res] = mod.reshape(mod.shape[0], res * res, 1)

    _MAP_CA_STORE = store
