"""
EasyControl nodes for Anima in ComfyUI.

LoadEasyControl: loads the adapter_model.safetensors into AnimaControlSelfAttn processors
ApplyEasyControlCondition: patches the model to inject condition tokens via self-attention LoRA
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import load_file as load_safetensors
from comfy.ldm.cosmos.predict2 import apply_rotary_pos_emb as cosmos_apply_rope
import comfy.ldm.common_dit

import comfy.model_patcher
import comfy.patcher_extension
from comfy.patcher_extension import WrappersMP


# ============================================================================
# EasyControl LoRA modules (same as diffusion-pipe/models/easycontrol.py)
# ============================================================================

class AnimaLoRALinearLayer(nn.Module):
    """LoRA layer with binary masking that only affects condition tokens."""
    def __init__(self, in_features, out_features, rank=4, network_alpha=None,
                 cond_size=1024, number=0, n_loras=1):
        super().__init__()
        self.rank = rank
        self.network_alpha = network_alpha
        self.cond_size = cond_size
        self.number = number
        self.n_loras = n_loras
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.down.weight, std=1.0 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states, cond_size=None):
        B, seq_len, _ = hidden_states.shape
        cs = cond_size if cond_size is not None else self.cond_size
        noise_len = seq_len - cs * self.n_loras
        mask = torch.zeros(B, seq_len, 1, device=hidden_states.device, dtype=hidden_states.dtype)
        cond_start = noise_len + self.number * cs
        mask[:, cond_start:cond_start + cs, :] = 1.0
        hidden_states = hidden_states * mask
        out = self.up(self.down(hidden_states))
        if self.network_alpha is not None:
            out = out * (self.network_alpha / self.rank)
        return out


def build_causal_attn_mask(noise_len, cond_size, n_conds, device, dtype, cond_attn_scale=1.0):
    import math
    total_len = noise_len + cond_size * n_conds
    neg_inf = -65504.0 if dtype == torch.float16 else -1e20
    mask = torch.full((total_len, total_len), neg_inf, device=device, dtype=dtype)
    # Noise → noise: full attention
    mask[:noise_len, :noise_len] = 0.0
    # Noise → cond: scaled attention (log(scale) bias in logit space)
    if cond_attn_scale > 1e-6:
        cond_bias = math.log(cond_attn_scale)  # 1.0→0, 0.5→-0.693, 0.1→-2.3
        cond_start = noise_len
        mask[:noise_len, cond_start:] = cond_bias
    # Cond → cond: each block sees only itself
    for i in range(n_conds):
        s = noise_len + i * cond_size
        mask[s:s + cond_size, s:s + cond_size] = 0.0
    return mask.unsqueeze(0).unsqueeze(0)


class AnimaControlSelfAttn(nn.Module):
    """Self-attention with LoRA injection and KV cache for condition tokens.

    On the first step (use_cache=False): full attention with condition tokens,
    LoRA, and causal mask. Caches K/V for condition tokens.
    On subsequent steps (use_cache=True): noise-only attention using cached K/V,
    NO LoRA, NO causal mask → much faster (58-75% reduction per paper).
    """
    def __init__(self, dim=2048, rank=128, network_alpha=128.0, cond_size=1024, n_loras=1):
        super().__init__()
        self.dim = dim
        self.cond_size = cond_size
        self.n_loras = n_loras
        for name in ['q_loras', 'k_loras', 'v_loras', 'proj_loras']:
            setattr(self, name, nn.ModuleList([
                AnimaLoRALinearLayer(dim, dim, rank, network_alpha, cond_size, i, n_loras)
                for i in range(n_loras)
            ]))
        # KV cache (populated on first step)
        self.cached_k = None
        self.cached_v = None
        self.cached_attn_out = None

    def clear_cache(self):
        self.cached_k = None
        self.cached_v = None
        self.cached_attn_out = None

    def forward(self, base_attn, noise_flat, cond_flat, noise_rope, cond_rope,
                lora_weights=None, use_cache=False, cond_attn_scale=1.0):

        if lora_weights is None:
            lora_weights = [1.0] * self.n_loras
        B, L_noise, D = noise_flat.shape
        n_heads = base_attn.n_heads
        head_dim = base_attn.head_dim

        if use_cache and self.cached_k is not None:
            # CACHED PATH: only compute Q for noise, use cached K/V from condition
            q = base_attn.q_proj(noise_flat)
            q = q.view(B, L_noise, n_heads, head_dim)
            q = base_attn.q_norm(q)
            if base_attn.is_selfattn:
                noise_rope_emb = noise_rope.unsqueeze(0).unsqueeze(2)
                q = cosmos_apply_rope(q, noise_rope_emb)
            q = q.transpose(1, 2)  # (B, H, L_noise, D)

            # Noise K/V (no LoRA on cached path)
            k_noise = base_attn.k_proj(noise_flat).view(B, L_noise, n_heads, head_dim)
            v_noise = base_attn.v_proj(noise_flat).view(B, L_noise, n_heads, head_dim)
            k_noise = base_attn.k_norm(k_noise)
            if base_attn.is_selfattn:
                k_noise = cosmos_apply_rope(k_noise, noise_rope_emb)
            k_noise = k_noise.transpose(1, 2)
            v_noise = v_noise.transpose(1, 2)

            # Expand cached K/V to batch size (CFG may change B between steps)
            ck = self.cached_k.expand(B, -1, -1, -1)
            cv = self.cached_v.expand(B, -1, -1, -1)

            # Concatenate noise + cached condition K/V
            k = torch.cat([k_noise, ck], dim=2)
            v = torch.cat([v_noise, cv], dim=2)

            if q.dtype != v.dtype:
                q = q.to(v.dtype)
                k = k.to(v.dtype)

            # Standard attention (no causal mask needed — noise queries see everything)
            x = F.scaled_dot_product_attention(q, k, v)
            del q, k, v

            x = x.transpose(1, 2).reshape(B, L_noise, n_heads * head_dim)
            noise_out = base_attn.output_dropout(base_attn.output_proj(x))
            del x

            # Condition output from cache
            cond_out = self.cached_attn_out.expand(B, -1, -1)
            return noise_out, cond_out

        # FULL PATH (first step): compute everything with LoRA and cache
        if cond_flat is None:
            # No condition data — run noise-only attention as fallback
            q = base_attn.q_proj(noise_flat).view(B, L_noise, n_heads, head_dim)
            q = base_attn.q_norm(q)
            if base_attn.is_selfattn and noise_rope is not None:
                q = cosmos_apply_rope(q, noise_rope.unsqueeze(0).unsqueeze(2))
            k = base_attn.k_proj(noise_flat).view(B, L_noise, n_heads, head_dim)
            k = base_attn.k_norm(k)
            if base_attn.is_selfattn and noise_rope is not None:
                k = cosmos_apply_rope(k, noise_rope.unsqueeze(0).unsqueeze(2))
            v = base_attn.v_proj(noise_flat).view(B, L_noise, n_heads, head_dim)
            x = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2))
            x = x.transpose(1,2).reshape(B, L_noise, n_heads * head_dim)
            noise_out = base_attn.output_dropout(base_attn.output_proj(x))
            return noise_out, torch.zeros(B, 0, noise_out.shape[-1], device=noise_out.device, dtype=noise_out.dtype)

        L_cond = cond_flat.shape[1]
        joint = torch.cat([noise_flat, cond_flat], dim=1)
        joint_rope = torch.cat([noise_rope, cond_rope], dim=0)

        q = base_attn.q_proj(joint)
        k = base_attn.k_proj(joint)
        v = base_attn.v_proj(joint)

        for i in range(self.n_loras):
            w = lora_weights[i]
            if w != 0.0:
                q = q + w * self.q_loras[i](joint, cond_size=L_cond)
                k = k + w * self.k_loras[i](joint, cond_size=L_cond)
                v = v + w * self.v_loras[i](joint, cond_size=L_cond)

        q = q.view(B, -1, n_heads, head_dim)
        k = k.view(B, -1, n_heads, head_dim)
        v = v.view(B, -1, n_heads, head_dim)

        q = base_attn.q_norm(q)
        k = base_attn.k_norm(k)

        if base_attn.is_selfattn:
            rope_for_attn = joint_rope.unsqueeze(0).unsqueeze(2)
            q = cosmos_apply_rope(q, rope_for_attn)
            k = cosmos_apply_rope(k, rope_for_attn)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if q.dtype != v.dtype:
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        causal_mask = build_causal_attn_mask(L_noise, L_cond, self.n_loras, q.device, q.dtype,
                                                   cond_attn_scale=cond_attn_scale)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=causal_mask)

        # Cache condition K/V (after norm+RoPE, in SDPA format: B, H, L, D)
        # Store only batch=1 slice to save memory (expanded at use time)
        self.cached_k = k[:1, :, L_noise:, :].contiguous()
        self.cached_v = v[:1, :, L_noise:, :].contiguous()

        del q, k, v, causal_mask

        x = x.transpose(1, 2).reshape(B, -1, n_heads * head_dim)

        out = base_attn.output_dropout(base_attn.output_proj(x))
        for i in range(self.n_loras):
            w = lora_weights[i]
            if w != 0.0:
                out = out + w * self.proj_loras[i](x, cond_size=L_cond)
        del x

        # Cache condition attention output (for residual stream)
        self.cached_attn_out = out[:1, L_noise:, :].contiguous()

        return out[:, :L_noise, :], out[:, L_noise:, :]


# ============================================================================
# RoPE generation for condition tokens
# ============================================================================

def generate_condition_rope(pos_embedder, noise_hw, cond_hw, device):
    """Generate RoPE with interpolated positions for condition tokens.

    Produces the SAME format as ComfyUI's VideoRopePosition3DEmb.generate_embeddings:
    shape (L, D, 2, 2) where the last dims are rotation matrix [cos, -sin, sin, cos].
    """
    from einops import repeat as einops_repeat

    H_noise, W_noise = noise_hw
    H_cond, W_cond = cond_hw
    T = 1

    h_theta = 10000.0 * pos_embedder.h_ntk_factor
    w_theta = 10000.0 * pos_embedder.w_ntk_factor
    t_theta = 10000.0 * pos_embedder.t_ntk_factor

    h_freqs = 1.0 / (h_theta ** pos_embedder.dim_spatial_range.to(device))
    w_freqs = 1.0 / (w_theta ** pos_embedder.dim_spatial_range.to(device))
    t_freqs = 1.0 / (t_theta ** pos_embedder.dim_temporal_range.to(device))

    frac_h = torch.linspace(0, H_noise - 1, H_cond, device=device) if H_cond > 1 else torch.zeros(1, device=device)
    frac_w = torch.linspace(0, W_noise - 1, W_cond, device=device) if W_cond > 1 else torch.zeros(1, device=device)
    frac_t = torch.zeros(T, device=device)

    half_h = torch.outer(frac_h, h_freqs)
    half_w = torch.outer(frac_w, w_freqs)
    half_t = torch.outer(frac_t, t_freqs)

    # Build rotation matrices [cos, -sin, sin, cos] — same as ComfyUI's generate_embeddings
    half_emb_h = torch.stack([torch.cos(half_h), -torch.sin(half_h), torch.sin(half_h), torch.cos(half_h)], dim=-1)
    half_emb_w = torch.stack([torch.cos(half_w), -torch.sin(half_w), torch.sin(half_w), torch.cos(half_w)], dim=-1)
    half_emb_t = torch.stack([torch.cos(half_t), -torch.sin(half_t), torch.sin(half_t), torch.cos(half_t)], dim=-1)

    em_T_H_W_D = torch.cat([
        einops_repeat(half_emb_t, "t d x -> t h w d x", h=H_cond, w=W_cond),
        einops_repeat(half_emb_h, "h d x -> t h w d x", t=T, w=W_cond),
        einops_repeat(half_emb_w, "w d x -> t h w d x", t=T, h=H_cond),
    ], dim=-2)

    # Final shape: (L, D, 2, 2) — matching ComfyUI format
    return rearrange(em_T_H_W_D, "t h w d (i j) -> (t h w) d i j", i=2, j=2).float()


def compute_condition_t_embedding(t_embedder, t_embedding_norm, batch_size, device, dtype):
    t_zero = torch.zeros(batch_size, 1, device=device, dtype=dtype)
    emb, adaln_lora = t_embedder[1](t_embedder[0](t_zero).to(dtype))
    emb = t_embedding_norm(emb)
    return emb, adaln_lora


# ============================================================================
# ComfyUI Nodes
# ============================================================================

class LoadEasyControl:
    """Load an EasyControl LoRA adapter for Anima."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "adapter_path": ("STRING", {"default": "", "tooltip": "Path to adapter_model.safetensors"}),
            }
        }

    RETURN_TYPES = ("EASYCONTROL_MODEL",)
    RETURN_NAMES = ("easycontrol_model",)
    FUNCTION = "load"
    CATEGORY = "conditioning/easycontrol"

    def load(self, adapter_path):
        sd = load_safetensors(adapter_path, device="cpu")

        # Infer rank
        rank = 128
        for k, v in sd.items():
            if '.down.weight' in k:
                rank = v.shape[0]
                break

        # Read metadata
        alpha = float(rank)
        n_loras = 1
        try:
            from safetensors import safe_open
            with safe_open(adapter_path, framework="pt") as f:
                meta = f.metadata() or {}
                rank = int(meta.get('rank', rank))
                alpha = float(meta.get('network_alpha', meta.get('alpha', alpha)))
                n_loras = int(meta.get('n_loras', n_loras))
        except Exception:
            pass

        # We don't know dim/cond_size yet — will be determined at apply time
        easycontrol_data = {
            "state_dict": sd,
            "rank": rank,
            "alpha": alpha,
            "n_loras": n_loras,
            "path": adapter_path,
        }
        return (easycontrol_data,)


class ApplyEasyControlCondition:
    """Apply a spatial condition (canny, depth, etc.) to an Anima model using EasyControl.

    Can be chained: each apply adds one condition slot.
    """

    # Cache processors to avoid recreating them when only strength changes
    # (similar to how LoraLoader caches loaded lora weights)
    _cached_processors = None  # (adapter_path, processors)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "easycontrol_model": ("EASYCONTROL_MODEL",),
                "control_latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/easycontrol"

    def apply(self, model, easycontrol_model, control_latent, strength):
        # Clone model patcher (non-destructive)
        model_clone = model.clone()

        # Get model info
        diffusion_model = model_clone.model.diffusion_model
        model_channels = diffusion_model.model_channels
        num_blocks = len(diffusion_model.blocks)

        # Build processors if not already attached
        existing = model_clone.model_options.get("easycontrol_conditions", [])

        # Reuse cached processors if same adapter (avoids recreating on strength change)
        adapter_path = easycontrol_model.get("path", "")
        rank = easycontrol_model["rank"]
        alpha = easycontrol_model["alpha"]
        n_loras = easycontrol_model["n_loras"]

        if (ApplyEasyControlCondition._cached_processors is not None
                and ApplyEasyControlCondition._cached_processors[0] == adapter_path):
            processors = ApplyEasyControlCondition._cached_processors[1]
            # Clear KV cache when reusing (strength changed → fresh generation)
            for proc in processors:
                proc.clear_cache()
        else:
            # Create new processors
            sd = easycontrol_model["state_dict"]
            processors = nn.ModuleList([
                AnimaControlSelfAttn(dim=model_channels, rank=rank, network_alpha=alpha,
                                     cond_size=1024, n_loras=n_loras)
                for _ in range(num_blocks)
            ])
            clean_sd = {}
            for k, v in sd.items():
                clean_k = k.replace('control_processors.', '') if k.startswith('control_processors.') else k
                clean_sd[clean_k] = v
            processors.load_state_dict(clean_sd, strict=False)
            processors.eval()
            ApplyEasyControlCondition._cached_processors = (adapter_path, processors)

        # Use pre-encoded latent from standard VAE Encode node
        # ComfyUI LATENT dict has "samples" key: (B, C, H/8, W/8)
        control_latents = control_latent["samples"]  # (B, C, H/8, W/8)
        if control_latents.ndim == 4:
            control_latents = control_latents.unsqueeze(2)  # (B, C, 1, H/8, W/8)

        # CRITICAL: Normalize control latents the same way ComfyUI normalizes noise.
        # ComfyUI's sampler applies process_latent_in (mean-subtract + std-divide) to noise
        # before the denoising loop, but our control latent comes raw from VAE Encode.
        # Without this, the condition signal is on the wrong scale → weak/no effect.
        control_latents = model_clone.model.process_latent_in(control_latents)

        # Store this condition
        condition_entry = {
            "processors": processors,
            "control_latents": control_latents,
            "strength": strength,
        }
        new_conditions = existing + [condition_entry]
        model_clone.model_options["easycontrol_conditions"] = new_conditions

        # Always re-register wrapper to avoid stale closure from clone()
        _register_easycontrol_wrapper(model_clone)

        return (model_clone,)


def _register_easycontrol_wrapper(model_patcher):
    """Register the DIFFUSION_MODEL wrapper that replaces the block loop."""

    # Capture conditions at registration time (closure)
    _conditions_ref = model_patcher.model_options
    _step_counter = [-1.0]  # mutable closure: tracks last seen timestep for cache logic

    def easycontrol_diffusion_wrapper(executor, *args, **kwargs):
        """Wraps MiniTrainDIT._forward to inject EasyControl condition tokens."""
        conditions = _conditions_ref.get("easycontrol_conditions", [])

        if not conditions:
            # No conditions — run normally
            _step_counter[0] = -1.0
            return executor.execute(*args, **kwargs)

        # KV Cache: first call populates cache, subsequent calls reuse cached K/V.
        # Detect new generation: timestep jumps UP (flow matching goes high→low).
        # Use strict > to avoid clearing cache on same-timestep RK substeps.
        current_t = args[1].max().item()
        if current_t > _step_counter[0]:
            # New generation (timestep jumped up) → clear cache, full path
            _step_counter[0] = current_t
            use_cache = False
            for cond_entry in conditions:
                for proc in cond_entry["processors"]:
                    proc.clear_cache()
        else:
            # Same or lower timestep → use cache
            use_cache = (conditions[0]["processors"][0].cached_k is not None)
            _step_counter[0] = current_t

        # Unpack args: x, timesteps, context, fps, padding_mask, **kwargs
        x = args[0]  # (B, C, T, H, W)
        timesteps = args[1]
        context = args[2]
        fps = args[3] if len(args) > 3 else None
        padding_mask = args[4] if len(args) > 4 else None

        # Get the actual model (MiniTrainDIT)
        dit = executor.class_obj

        # Get transformer_options from kwargs
        transformer_options = kwargs.get("transformer_options", {})

        # Run the standard preamble (same as _forward)
        orig_shape = list(x.shape)
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (dit.patch_temporal, dit.patch_spatial, dit.patch_spatial))

        x_5d, rope_emb, extra_pos_emb = dit.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_emb, adaln_lora = dit.t_embedder[1](dit.t_embedder[0](timesteps).to(x_5d.dtype))
        t_emb = dit.t_embedding_norm(t_emb)

        B, T, H, W, D = x_5d.shape
        device = x_5d.device
        dtype = context.dtype

        if x_5d.dtype == torch.float16:
            x_5d = x_5d.float()

        # Prepare ALL conditions
        cond_states = []
        cond_ropes = []
        cond_processors_list = []
        cond_strengths = []

        # Move processors to device ONCE (not every step)
        for cond_entry in conditions:
            if not cond_entry.get("_on_device", False):
                cond_entry["processors"].to(device, dtype=dtype)
                cond_entry["control_latents"] = cond_entry["control_latents"].to(device, dtype=dtype)
                cond_entry["_on_device"] = True

        if not use_cache:
            # First step: prepare condition embeddings, t_emb, etc.
            cond_t_emb, cond_adaln_lora = compute_condition_t_embedding(
                dit.t_embedder, dit.t_embedding_norm, B, device, dtype
            )

            for cond_entry in conditions:
                processors = cond_entry["processors"]
                ctrl_latents = cond_entry["control_latents"]
                strength = cond_entry["strength"]

                if ctrl_latents.shape[0] < B:
                    ctrl_latents = ctrl_latents.expand(B, -1, -1, -1, -1) if ctrl_latents.ndim == 5 else ctrl_latents.expand(B, -1, -1, -1)

                # Pad control latents to patch size (same as noise gets via pad_to_patch_size)
                ctrl_padded = comfy.ldm.common_dit.pad_to_patch_size(
                    ctrl_latents, (dit.patch_temporal, dit.patch_spatial, dit.patch_spatial))
                pad_cond = torch.zeros(B, 1, 1, ctrl_padded.shape[3], ctrl_padded.shape[4],
                                       device=device, dtype=ctrl_padded.dtype)
                cond_with_mask = torch.cat([ctrl_padded, pad_cond], dim=1)
                cond_embedded = dit.x_embedder(cond_with_mask)
                _, Tc, Hc, Wc, _ = cond_embedded.shape

                cond_rope = generate_condition_rope(dit.pos_embedder, (H, W), (Hc, Wc), device)

                cond_states.append(cond_embedded)
                cond_ropes.append(cond_rope)
                cond_processors_list.append(processors)
                cond_strengths.append(strength)
        else:
            # Cached steps: only need processors (already on device) and strengths
            for cond_entry in conditions:
                cond_processors_list.append(cond_entry["processors"])
                cond_strengths.append(cond_entry["strength"])

        # RoPE format: ComfyUI uses unsqueeze(1).unsqueeze(0) on rope
        noise_rope = rope_emb  # already in correct format from prepare_embedded_sequence

        block_kwargs = {
            "rope_emb_L_1_1_D": noise_rope.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora,
            "extra_per_block_pos_emb": extra_pos_emb,
            "transformer_options": transformer_options,
        }

        def _r(t):
            return rearrange(t, "b t d -> b t 1 1 d")

        # Run block loop with EasyControl injection
        for bi, block in enumerate(dit.blocks):
            residual_dtype = x_5d.dtype
            compute_dtype = t_emb.dtype

            if extra_pos_emb is not None:
                x_5d = x_5d + extra_pos_emb

            # AdaLN for noise
            if block.use_adaln_lora:
                sh_sa, sc_sa, g_sa = (block.adaln_modulation_self_attn(t_emb) + adaln_lora).chunk(3, -1)
                sh_ca, sc_ca, g_ca = (block.adaln_modulation_cross_attn(t_emb) + adaln_lora).chunk(3, -1)
                sh_ml, sc_ml, g_ml = (block.adaln_modulation_mlp(t_emb) + adaln_lora).chunk(3, -1)
            else:
                sh_sa, sc_sa, g_sa = block.adaln_modulation_self_attn(t_emb).chunk(3, -1)
                sh_ca, sc_ca, g_ca = block.adaln_modulation_cross_attn(t_emb).chunk(3, -1)
                sh_ml, sc_ml, g_ml = block.adaln_modulation_mlp(t_emb).chunk(3, -1)

            # --- Self-attention with ALL conditions ---
            norm_noise = block.layer_norm_self_attn(x_5d) * (1 + _r(sc_sa)) + _r(sh_sa)
            noise_flat = rearrange(norm_noise.to(compute_dtype), "b t h w d -> b (t h w) d")

            if use_cache:
                # CACHED PATH: noise-only attention with cached condition K/V
                for ci, (processors, strength) in enumerate(
                    zip(cond_processors_list, cond_strengths)
                ):
                    noise_out, cond_out = processors[bi](
                        block.self_attn, noise_flat, None, noise_rope, None,
                        lora_weights=[strength], use_cache=True,
                    )
                    noise_attn = rearrange(noise_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                    x_5d = x_5d + _r(g_sa).to(residual_dtype) * noise_attn.to(residual_dtype)
            else:
                # FULL PATH (first step): LoRA + causal mask + populate cache
                for ci, (cond_state, cond_rope, processors, strength) in enumerate(
                    zip(cond_states, cond_ropes, cond_processors_list, cond_strengths)
                ):
                    if block.use_adaln_lora:
                        c_sh_sa, c_sc_sa, c_g_sa = (block.adaln_modulation_self_attn(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                        c_sh_ml, c_sc_ml, c_g_ml = (block.adaln_modulation_mlp(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                    else:
                        c_sh_sa, c_sc_sa, c_g_sa = block.adaln_modulation_self_attn(cond_t_emb).chunk(3, -1)
                        c_sh_ml, c_sc_ml, c_g_ml = block.adaln_modulation_mlp(cond_t_emb).chunk(3, -1)

                    norm_cond = block.layer_norm_self_attn(cond_state) * (1 + _r(c_sc_sa)) + _r(c_sh_sa)
                    cond_flat = rearrange(norm_cond.to(compute_dtype), "b t hc wc d -> b (t hc wc) d")

                    _, Tc, Hc, Wc, _ = cond_state.shape

                    noise_out, cond_out = processors[bi](
                        block.self_attn, noise_flat, cond_flat, noise_rope, cond_rope,
                        lora_weights=[strength], use_cache=False,
                    )

                    noise_attn = rearrange(noise_out, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                    cond_attn = rearrange(cond_out, "b (t hc wc) d -> b t hc wc d", t=Tc, hc=Hc, wc=Wc)

                    x_5d = x_5d + _r(g_sa).to(residual_dtype) * noise_attn.to(residual_dtype)
                    cond_states[ci] = cond_state + _r(c_g_sa).to(residual_dtype) * cond_attn.to(residual_dtype)

                    if ci < len(cond_states) - 1:
                        norm_noise = block.layer_norm_self_attn(x_5d) * (1 + _r(sc_sa)) + _r(sh_sa)
                        noise_flat = rearrange(norm_noise.to(compute_dtype), "b t h w d -> b (t h w) d")

            # --- Cross-attention (noise only) ---
            norm_ca = block.layer_norm_cross_attn(x_5d) * (1 + _r(sc_ca)) + _r(sh_ca)
            ca_out = rearrange(
                block.cross_attn(
                    rearrange(norm_ca.to(compute_dtype), "b t h w d -> b (t h w) d"),
                    context,
                    rope_emb=block_kwargs["rope_emb_L_1_1_D"],
                    transformer_options=transformer_options,
                ),
                "b (t h w) d -> b t h w d", t=T, h=H, w=W,
            )
            x_5d = x_5d + _r(g_ca).to(residual_dtype) * ca_out.to(residual_dtype)

            # --- MLP (noise only on cached steps, noise+cond on first step) ---
            norm_ml = block.layer_norm_mlp(x_5d) * (1 + _r(sc_ml)) + _r(sh_ml)
            x_5d = x_5d + _r(g_ml).to(residual_dtype) * block.mlp(norm_ml.to(compute_dtype)).to(residual_dtype)

            if not use_cache:
                for ci, cond_state in enumerate(cond_states):
                    if block.use_adaln_lora:
                        c_sh_ml, c_sc_ml, c_g_ml = (block.adaln_modulation_mlp(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                    else:
                        c_sh_ml, c_sc_ml, c_g_ml = block.adaln_modulation_mlp(cond_t_emb).chunk(3, -1)
                    c_norm_ml = block.layer_norm_mlp(cond_state) * (1 + _r(c_sc_ml)) + _r(c_sh_ml)
                    cond_states[ci] = cond_state + _r(c_g_ml).to(residual_dtype) * block.mlp(c_norm_ml.to(compute_dtype)).to(residual_dtype)

        # Final layer (noise only)
        x_out = dit.final_layer(x_5d.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln_lora)
        result = dit.unpatchify(x_out)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]
        return result

    # Register the wrapper (is_model_options=True because we pass model_options, not transformer_options)
    comfy.patcher_extension.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "easycontrol_anima",
        easycontrol_diffusion_wrapper,
        model_patcher.model_options,
        is_model_options=True,
    )


# ============================================================================
# EasyControl Plus — LoRA-compatible variant
# ============================================================================

class ApplyEasyControlConditionPlus:
    """EasyControl Plus: LoRA-compatible spatial conditioning for Anima.

    Unlike the standard node, this runs the standard self-attention FIRST
    (preserving other LoRA effects), then blends in EasyControl influence.
    This prevents EasyControl from diluting style/character LoRAs.

    Trade-off: slightly slower (runs self-attention twice per block on step 1),
    but preserves full LoRA compatibility.
    """

    _cached_processors = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "easycontrol_model": ("EASYCONTROL_MODEL",),
                "control_latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/easycontrol"

    def apply(self, model, easycontrol_model, control_latent, strength):
        model_clone = model.clone()

        diffusion_model = model_clone.model.diffusion_model
        model_channels = diffusion_model.model_channels
        num_blocks = len(diffusion_model.blocks)

        existing = model_clone.model_options.get("easycontrol_conditions_plus", [])

        adapter_path = easycontrol_model.get("path", "")
        rank = easycontrol_model["rank"]
        alpha = easycontrol_model["alpha"]
        n_loras = easycontrol_model["n_loras"]

        if (ApplyEasyControlConditionPlus._cached_processors is not None
                and ApplyEasyControlConditionPlus._cached_processors[0] == adapter_path):
            processors = ApplyEasyControlConditionPlus._cached_processors[1]
            for proc in processors:
                proc.clear_cache()
        else:
            sd = easycontrol_model["state_dict"]
            processors = nn.ModuleList([
                AnimaControlSelfAttn(dim=model_channels, rank=rank, network_alpha=alpha,
                                     cond_size=1024, n_loras=n_loras)
                for _ in range(num_blocks)
            ])
            clean_sd = {}
            for k, v in sd.items():
                clean_k = k.replace('control_processors.', '') if k.startswith('control_processors.') else k
                clean_sd[clean_k] = v
            processors.load_state_dict(clean_sd, strict=False)
            processors.eval()
            ApplyEasyControlConditionPlus._cached_processors = (adapter_path, processors)

        control_latents = control_latent["samples"]
        if control_latents.ndim == 4:
            control_latents = control_latents.unsqueeze(2)
        control_latents = model_clone.model.process_latent_in(control_latents)

        condition_entry = {
            "processors": processors,
            "control_latents": control_latents,
            "strength": strength,
        }
        new_conditions = existing + [condition_entry]
        model_clone.model_options["easycontrol_conditions_plus"] = new_conditions

        _register_easycontrol_plus_wrapper(model_clone)

        return (model_clone,)


def _register_easycontrol_plus_wrapper(model_patcher):
    """EasyControl Plus wrapper: preserves LoRA by blending standard + EC attention."""

    _conditions_ref = model_patcher.model_options
    _step_counter = [-1.0]

    def easycontrol_plus_wrapper(executor, *args, **kwargs):
        conditions = _conditions_ref.get("easycontrol_conditions_plus", [])

        if not conditions:
            _step_counter[0] = -1.0
            return executor.execute(*args, **kwargs)

        current_t = args[1].max().item()
        if current_t > _step_counter[0]:
            _step_counter[0] = current_t
            use_cache = False
            for cond_entry in conditions:
                for proc in cond_entry["processors"]:
                    proc.clear_cache()
        else:
            use_cache = (conditions[0]["processors"][0].cached_k is not None)
            _step_counter[0] = current_t

        x = args[0]
        timesteps = args[1]
        context = args[2]
        fps = args[3] if len(args) > 3 else None
        padding_mask = args[4] if len(args) > 4 else None

        dit = executor.class_obj
        transformer_options = kwargs.get("transformer_options", {})

        orig_shape = list(x.shape)
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (dit.patch_temporal, dit.patch_spatial, dit.patch_spatial))

        x_5d, rope_emb, extra_pos_emb = dit.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_emb, adaln_lora = dit.t_embedder[1](dit.t_embedder[0](timesteps).to(x_5d.dtype))
        t_emb = dit.t_embedding_norm(t_emb)

        B, T, H, W, D = x_5d.shape
        device = x_5d.device
        dtype = context.dtype

        if x_5d.dtype == torch.float16:
            x_5d = x_5d.float()

        # Move processors once
        for cond_entry in conditions:
            if not cond_entry.get("_on_device", False):
                cond_entry["processors"].to(device, dtype=dtype)
                cond_entry["control_latents"] = cond_entry["control_latents"].to(device, dtype=dtype)
                cond_entry["_on_device"] = True

        # Prepare conditions (first step only)
        cond_states = []
        cond_ropes = []
        cond_processors_list = []
        cond_strengths = []

        if not use_cache:
            cond_t_emb, cond_adaln_lora = compute_condition_t_embedding(
                dit.t_embedder, dit.t_embedding_norm, B, device, dtype
            )
            for cond_entry in conditions:
                processors = cond_entry["processors"]
                ctrl_latents = cond_entry["control_latents"]
                strength = cond_entry["strength"]
                if ctrl_latents.shape[0] < B:
                    ctrl_latents = ctrl_latents.expand(B, -1, -1, -1, -1)
                # Pad control latents to patch size (same as noise gets via pad_to_patch_size)
                ctrl_padded = comfy.ldm.common_dit.pad_to_patch_size(
                    ctrl_latents, (dit.patch_temporal, dit.patch_spatial, dit.patch_spatial))
                pad_cond = torch.zeros(B, 1, 1, ctrl_padded.shape[3], ctrl_padded.shape[4],
                                       device=device, dtype=ctrl_padded.dtype)
                cond_with_mask = torch.cat([ctrl_padded, pad_cond], dim=1)
                cond_embedded = dit.x_embedder(cond_with_mask)
                _, Tc, Hc, Wc, _ = cond_embedded.shape
                cond_rope = generate_condition_rope(dit.pos_embedder, (H, W), (Hc, Wc), device)
                cond_states.append(cond_embedded)
                cond_ropes.append(cond_rope)
                cond_processors_list.append(processors)
                cond_strengths.append(strength)
        else:
            for cond_entry in conditions:
                cond_processors_list.append(cond_entry["processors"])
                cond_strengths.append(cond_entry["strength"])

        noise_rope = rope_emb

        block_kwargs = {
            "rope_emb_L_1_1_D": noise_rope.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora,
            "extra_per_block_pos_emb": extra_pos_emb,
            "transformer_options": transformer_options,
        }

        def _r(t):
            return rearrange(t, "b t d -> b t 1 1 d")

        for bi, block in enumerate(dit.blocks):
            residual_dtype = x_5d.dtype
            compute_dtype = t_emb.dtype

            if extra_pos_emb is not None:
                x_5d = x_5d + extra_pos_emb

            # AdaLN for noise
            if block.use_adaln_lora:
                sh_sa, sc_sa, g_sa = (block.adaln_modulation_self_attn(t_emb) + adaln_lora).chunk(3, -1)
                sh_ca, sc_ca, g_ca = (block.adaln_modulation_cross_attn(t_emb) + adaln_lora).chunk(3, -1)
                sh_ml, sc_ml, g_ml = (block.adaln_modulation_mlp(t_emb) + adaln_lora).chunk(3, -1)
            else:
                sh_sa, sc_sa, g_sa = block.adaln_modulation_self_attn(t_emb).chunk(3, -1)
                sh_ca, sc_ca, g_ca = block.adaln_modulation_cross_attn(t_emb).chunk(3, -1)
                sh_ml, sc_ml, g_ml = block.adaln_modulation_mlp(t_emb).chunk(3, -1)

            # --- STANDARD self-attention (preserves LoRA effects) ---
            norm_noise = block.layer_norm_self_attn(x_5d) * (1 + _r(sc_sa)) + _r(sh_sa)
            noise_flat = rearrange(norm_noise.to(compute_dtype), "b t h w d -> b (t h w) d")

            # Run standard self-attention through ComfyUI's full path (LoRA-aware)
            noise_std = block.self_attn(
                noise_flat, None,
                rope_emb=block_kwargs["rope_emb_L_1_1_D"],
                transformer_options=transformer_options,
            )

            if use_cache:
                # Cached: blend standard + cached EasyControl
                for ci, (processors, strength) in enumerate(
                    zip(cond_processors_list, cond_strengths)
                ):
                    noise_ec, _ = processors[bi](
                        block.self_attn, noise_flat, None, noise_rope, None,
                        lora_weights=[strength], use_cache=True,
                    )
                    # Blend: (1-s)*standard + s*easycontrol
                    noise_blended = noise_std + strength * (noise_ec - noise_std)
                    noise_attn = rearrange(noise_blended, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                    x_5d = x_5d + _r(g_sa).to(residual_dtype) * noise_attn.to(residual_dtype)
            else:
                # Full path: run EasyControl, blend, update condition state
                for ci, (cond_state, cond_rope, processors, strength) in enumerate(
                    zip(cond_states, cond_ropes, cond_processors_list, cond_strengths)
                ):
                    if block.use_adaln_lora:
                        c_sh_sa, c_sc_sa, c_g_sa = (block.adaln_modulation_self_attn(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                        c_sh_ml, c_sc_ml, c_g_ml = (block.adaln_modulation_mlp(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                    else:
                        c_sh_sa, c_sc_sa, c_g_sa = block.adaln_modulation_self_attn(cond_t_emb).chunk(3, -1)
                        c_sh_ml, c_sc_ml, c_g_ml = block.adaln_modulation_mlp(cond_t_emb).chunk(3, -1)

                    norm_cond = block.layer_norm_self_attn(cond_state) * (1 + _r(c_sc_sa)) + _r(c_sh_sa)
                    cond_flat = rearrange(norm_cond.to(compute_dtype), "b t hc wc d -> b (t hc wc) d")
                    _, Tc, Hc, Wc, _ = cond_state.shape

                    noise_ec, cond_out = processors[bi](
                        block.self_attn, noise_flat, cond_flat, noise_rope, cond_rope,
                        lora_weights=[strength], use_cache=False,
                    )

                    # Blend: (1-s)*standard + s*easycontrol
                    noise_blended = noise_std + strength * (noise_ec - noise_std)

                    noise_attn = rearrange(noise_blended, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                    cond_attn = rearrange(cond_out, "b (t hc wc) d -> b t hc wc d", t=Tc, hc=Hc, wc=Wc)

                    x_5d = x_5d + _r(g_sa).to(residual_dtype) * noise_attn.to(residual_dtype)
                    cond_states[ci] = cond_state + _r(c_g_sa).to(residual_dtype) * cond_attn.to(residual_dtype)

                    if ci < len(cond_states) - 1:
                        norm_noise = block.layer_norm_self_attn(x_5d) * (1 + _r(sc_sa)) + _r(sh_sa)
                        noise_flat = rearrange(norm_noise.to(compute_dtype), "b t h w d -> b (t h w) d")

            # --- Cross-attention (standard path) ---
            norm_ca = block.layer_norm_cross_attn(x_5d) * (1 + _r(sc_ca)) + _r(sh_ca)
            ca_out = rearrange(
                block.cross_attn(
                    rearrange(norm_ca.to(compute_dtype), "b t h w d -> b (t h w) d"),
                    context,
                    rope_emb=block_kwargs["rope_emb_L_1_1_D"],
                    transformer_options=transformer_options,
                ),
                "b (t h w) d -> b t h w d", t=T, h=H, w=W,
            )
            x_5d = x_5d + _r(g_ca).to(residual_dtype) * ca_out.to(residual_dtype)

            # --- MLP ---
            norm_ml = block.layer_norm_mlp(x_5d) * (1 + _r(sc_ml)) + _r(sh_ml)
            x_5d = x_5d + _r(g_ml).to(residual_dtype) * block.mlp(norm_ml.to(compute_dtype)).to(residual_dtype)

            if not use_cache:
                for ci, cond_state in enumerate(cond_states):
                    if block.use_adaln_lora:
                        c_sh_ml, c_sc_ml, c_g_ml = (block.adaln_modulation_mlp(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                    else:
                        c_sh_ml, c_sc_ml, c_g_ml = block.adaln_modulation_mlp(cond_t_emb).chunk(3, -1)
                    c_norm_ml = block.layer_norm_mlp(cond_state) * (1 + _r(c_sc_ml)) + _r(c_sh_ml)
                    cond_states[ci] = cond_state + _r(c_g_ml).to(residual_dtype) * block.mlp(c_norm_ml.to(compute_dtype)).to(residual_dtype)

        # Final layer
        x_out = dit.final_layer(x_5d.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln_lora)
        result = dit.unpatchify(x_out)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]
        return result

    comfy.patcher_extension.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "easycontrol_anima_plus",
        easycontrol_plus_wrapper,
        model_patcher.model_options,
        is_model_options=True,
    )


# ============================================================================
# EasyControl Experimental — Optimized Plus with shared projections
# ============================================================================

class ApplyEasyControlExperimental:
    """EasyControl Experimental: Step scheduling + block selection.

    Controls WHEN and WHERE EasyControl is active:
    - start/end_percent: EC only runs on the first N% of steps (e.g., 0.0-0.3 = first 30%)
      Steps after end_percent use pure standard attention → full LoRA effect
    - start/end_block: EC only on specific transformer blocks (0-27)
    - No overhead on inactive steps (standard _forward runs directly)
    """

    _cached_processors = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "easycontrol_model": ("EASYCONTROL_MODEL",),
                "control_latent": ("LATENT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Start applying EC at this % of sampling (0.0 = from start)"}),
                "end_percent": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Stop applying EC at this % of sampling (0.5 = first 50% of steps)"}),
                "start_block": ("INT", {"default": 0, "min": 0, "max": 27, "step": 1,
                                        "tooltip": "First transformer block to apply EC"}),
                "end_block": ("INT", {"default": 27, "min": 0, "max": 27, "step": 1,
                                      "tooltip": "Last transformer block to apply EC"}),
                "cond_attn_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                              "tooltip": "How much attention flows to condition tokens. 1.0=full EC, 0.5=half (preserves LoRA style), 0.0=EC off"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/easycontrol"

    def apply(self, model, easycontrol_model, control_latent, strength, start_percent, end_percent, start_block, end_block, cond_attn_scale):
        model_clone = model.clone()
        diffusion_model = model_clone.model.diffusion_model
        model_channels = diffusion_model.model_channels
        num_blocks = len(diffusion_model.blocks)

        existing = model_clone.model_options.get("easycontrol_conditions_exp", [])

        adapter_path = easycontrol_model.get("path", "")
        rank = easycontrol_model["rank"]
        alpha = easycontrol_model["alpha"]
        n_loras = easycontrol_model["n_loras"]

        if (ApplyEasyControlExperimental._cached_processors is not None
                and ApplyEasyControlExperimental._cached_processors[0] == adapter_path):
            processors = ApplyEasyControlExperimental._cached_processors[1]
            for proc in processors:
                proc.clear_cache()
        else:
            sd = easycontrol_model["state_dict"]
            processors = nn.ModuleList([
                AnimaControlSelfAttn(dim=model_channels, rank=rank, network_alpha=alpha,
                                     cond_size=1024, n_loras=n_loras)
                for _ in range(num_blocks)
            ])
            clean_sd = {}
            for k, v in sd.items():
                clean_k = k.replace('control_processors.', '') if k.startswith('control_processors.') else k
                clean_sd[clean_k] = v
            processors.load_state_dict(clean_sd, strict=False)
            processors.eval()
            ApplyEasyControlExperimental._cached_processors = (adapter_path, processors)

        control_latents = control_latent["samples"]
        if control_latents.ndim == 4:
            control_latents = control_latents.unsqueeze(2)
        control_latents = model_clone.model.process_latent_in(control_latents)

        condition_entry = {
            "processors": processors,
            "control_latents": control_latents,
            "strength": strength,
            "start_percent": start_percent,
            "end_percent": end_percent,
            "start_block": start_block,
            "end_block": end_block,
            "cond_attn_scale": cond_attn_scale,
        }
        new_conditions = existing + [condition_entry]
        model_clone.model_options["easycontrol_conditions_exp"] = new_conditions

        _register_easycontrol_exp_wrapper(model_clone)

        return (model_clone,)


def _register_easycontrol_exp_wrapper(model_patcher):
    """Experimental wrapper: step scheduling + block selection."""

    _conditions_ref = model_patcher.model_options
    _step_counter = [-1.0]
    _max_t = [1.0]  # tracks the highest timestep seen (= sigma_max)
    _step_num = [0]  # tracks current step number for progress calculation

    def easycontrol_exp_wrapper(executor, *args, **kwargs):
        conditions = _conditions_ref.get("easycontrol_conditions_exp", [])

        if not conditions:
            _step_counter[0] = -1.0
            _step_num[0] = 0
            return executor(*args, **kwargs)

        current_t = args[1].max().item()
        if current_t > _step_counter[0]:
            # New generation
            _step_counter[0] = current_t
            _max_t[0] = current_t
            _step_num[0] = 0
            use_cache = False
            for cond_entry in conditions:
                for proc in cond_entry["processors"]:
                    proc.clear_cache()
        else:
            use_cache = (conditions[0]["processors"][0].cached_k is not None)
            if current_t < _step_counter[0]:
                _step_num[0] += 1
            _step_counter[0] = current_t

        # Calculate sampling progress (0.0 = start, 1.0 = end)
        # Flow matching: timesteps go from max_t → 0. Progress = 1 - (current/max).
        progress = 1.0 - (current_t / _max_t[0]) if _max_t[0] > 0 else 0.0

        # Check if ANY condition is active at this progress
        any_step_active = any(
            cond.get("start_percent", 0.0) <= progress <= cond.get("end_percent", 1.0)
            for cond in conditions
        )

        if not any_step_active:
            # EC is OFF for this step — continue chain (other wrappers like LoRA still work)
            return executor(*args, **kwargs)

        x = args[0]
        timesteps = args[1]
        context = args[2]
        fps = args[3] if len(args) > 3 else None
        padding_mask = args[4] if len(args) > 4 else None

        dit = executor.class_obj
        transformer_options = kwargs.get("transformer_options", {})

        orig_shape = list(x.shape)
        x = comfy.ldm.common_dit.pad_to_patch_size(x, (dit.patch_temporal, dit.patch_spatial, dit.patch_spatial))
        x_5d, rope_emb, extra_pos_emb = dit.prepare_embedded_sequence(x, fps=fps, padding_mask=padding_mask)

        if timesteps.ndim == 1:
            timesteps = timesteps.unsqueeze(1)
        t_emb, adaln_lora = dit.t_embedder[1](dit.t_embedder[0](timesteps).to(x_5d.dtype))
        t_emb = dit.t_embedding_norm(t_emb)

        B, T, H, W, D = x_5d.shape
        device = x_5d.device
        dtype = context.dtype

        if x_5d.dtype == torch.float16:
            x_5d = x_5d.float()

        for cond_entry in conditions:
            if not cond_entry.get("_on_device", False):
                cond_entry["processors"].to(device, dtype=dtype)
                cond_entry["control_latents"] = cond_entry["control_latents"].to(device, dtype=dtype)
                cond_entry["_on_device"] = True

        cond_states = []
        cond_ropes = []
        cond_processors_list = []
        cond_strengths = []
        cond_block_ranges = []
        cond_attn_scales = []

        if not use_cache:
            cond_t_emb, cond_adaln_lora = compute_condition_t_embedding(
                dit.t_embedder, dit.t_embedding_norm, B, device, dtype
            )
            for cond_entry in conditions:
                processors = cond_entry["processors"]
                ctrl_latents = cond_entry["control_latents"]
                strength = cond_entry["strength"]
                if ctrl_latents.shape[0] < B:
                    ctrl_latents = ctrl_latents.expand(B, -1, -1, -1, -1)
                ctrl_padded = comfy.ldm.common_dit.pad_to_patch_size(
                    ctrl_latents, (dit.patch_temporal, dit.patch_spatial, dit.patch_spatial))
                pad_cond = torch.zeros(B, 1, 1, ctrl_padded.shape[3], ctrl_padded.shape[4],
                                       device=device, dtype=ctrl_padded.dtype)
                cond_with_mask = torch.cat([ctrl_padded, pad_cond], dim=1)
                cond_embedded = dit.x_embedder(cond_with_mask)
                _, Tc, Hc, Wc, _ = cond_embedded.shape
                cond_rope = generate_condition_rope(dit.pos_embedder, (H, W), (Hc, Wc), device)
                cond_states.append(cond_embedded)
                cond_ropes.append(cond_rope)
                cond_processors_list.append(processors)
                cond_strengths.append(strength)
                cond_block_ranges.append((cond_entry.get("start_block", 0), cond_entry.get("end_block", 27)))
                cond_attn_scales.append(cond_entry.get("cond_attn_scale", 1.0))
        else:
            for cond_entry in conditions:
                cond_processors_list.append(cond_entry["processors"])
                cond_strengths.append(cond_entry["strength"])
                cond_block_ranges.append((cond_entry.get("start_block", 0), cond_entry.get("end_block", 27)))
                cond_attn_scales.append(cond_entry.get("cond_attn_scale", 1.0))

        noise_rope = rope_emb
        block_kwargs = {
            "rope_emb_L_1_1_D": noise_rope.unsqueeze(1).unsqueeze(0),
            "adaln_lora_B_T_3D": adaln_lora,
            "extra_per_block_pos_emb": extra_pos_emb,
            "transformer_options": transformer_options,
        }

        def _r(t):
            return rearrange(t, "b t d -> b t 1 1 d")

        for bi, block in enumerate(dit.blocks):
            residual_dtype = x_5d.dtype
            compute_dtype = t_emb.dtype

            if extra_pos_emb is not None:
                x_5d = x_5d + extra_pos_emb

            if block.use_adaln_lora:
                sh_sa, sc_sa, g_sa = (block.adaln_modulation_self_attn(t_emb) + adaln_lora).chunk(3, -1)
                sh_ca, sc_ca, g_ca = (block.adaln_modulation_cross_attn(t_emb) + adaln_lora).chunk(3, -1)
                sh_ml, sc_ml, g_ml = (block.adaln_modulation_mlp(t_emb) + adaln_lora).chunk(3, -1)
            else:
                sh_sa, sc_sa, g_sa = block.adaln_modulation_self_attn(t_emb).chunk(3, -1)
                sh_ca, sc_ca, g_ca = block.adaln_modulation_cross_attn(t_emb).chunk(3, -1)
                sh_ml, sc_ml, g_ml = block.adaln_modulation_mlp(t_emb).chunk(3, -1)

            norm_noise = block.layer_norm_self_attn(x_5d) * (1 + _r(sc_sa)) + _r(sh_sa)
            noise_flat = rearrange(norm_noise.to(compute_dtype), "b t h w d -> b (t h w) d")

            # Check if any condition applies to this block
            any_active = any(s <= bi <= e for s, e in cond_block_ranges)

            if not any_active:
                # Standard self-attention (no EasyControl on this block)
                noise_std = block.self_attn(
                    noise_flat, None,
                    rope_emb=block_kwargs["rope_emb_L_1_1_D"],
                    transformer_options=transformer_options,
                )
                noise_attn = rearrange(noise_std, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                x_5d = x_5d + _r(g_sa).to(residual_dtype) * noise_attn.to(residual_dtype)
            else:
                # Standard noise self-attention (LoRA-preserved)
                noise_std = block.self_attn(
                    noise_flat, None,
                    rope_emb=block_kwargs["rope_emb_L_1_1_D"],
                    transformer_options=transformer_options,
                )

                if use_cache:
                    # OPTIMIZED CACHED PATH: share Q/K/V projections
                    # In cached mode, EC doesn't apply LoRA to noise, so Q/K/V are identical
                    # between standard and EC paths. We already have noise_std from above.
                    # Just need EC's cached result (which includes condition K/V influence).
                    for ci, (processors, strength) in enumerate(
                        zip(cond_processors_list, cond_strengths)
                    ):
                        s, e = cond_block_ranges[ci]
                        if not (s <= bi <= e):
                            continue
                        noise_ec, _ = processors[bi](
                            block.self_attn, noise_flat, None, noise_rope, None,
                            lora_weights=[strength], use_cache=True,
                            cond_attn_scale=cond_attn_scales[ci],
                        )
                        noise_blended = noise_std + strength * (noise_ec - noise_std)
                        noise_std = noise_blended

                    noise_attn = rearrange(noise_std, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                    x_5d = x_5d + _r(g_sa).to(residual_dtype) * noise_attn.to(residual_dtype)
                else:
                    # FULL PATH (step 1): standard + EC with full LoRA
                    for ci, (cond_state, cond_rope, processors, strength) in enumerate(
                        zip(cond_states, cond_ropes, cond_processors_list, cond_strengths)
                    ):
                        s, e = cond_block_ranges[ci]
                        if not (s <= bi <= e):
                            continue

                        if block.use_adaln_lora:
                            c_sh_sa, c_sc_sa, c_g_sa = (block.adaln_modulation_self_attn(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                            c_sh_ml, c_sc_ml, c_g_ml = (block.adaln_modulation_mlp(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                        else:
                            c_sh_sa, c_sc_sa, c_g_sa = block.adaln_modulation_self_attn(cond_t_emb).chunk(3, -1)
                            c_sh_ml, c_sc_ml, c_g_ml = block.adaln_modulation_mlp(cond_t_emb).chunk(3, -1)

                        norm_cond = block.layer_norm_self_attn(cond_state) * (1 + _r(c_sc_sa)) + _r(c_sh_sa)
                        cond_flat = rearrange(norm_cond.to(compute_dtype), "b t hc wc d -> b (t hc wc) d")
                        _, Tc, Hc, Wc, _ = cond_state.shape

                        noise_ec, cond_out = processors[bi](
                            block.self_attn, noise_flat, cond_flat, noise_rope, cond_rope,
                            lora_weights=[strength], use_cache=False,
                            cond_attn_scale=cond_attn_scales[ci],
                        )

                        noise_blended = noise_std + strength * (noise_ec - noise_std)
                        noise_std = noise_blended

                        cond_attn = rearrange(cond_out, "b (t hc wc) d -> b t hc wc d", t=Tc, hc=Hc, wc=Wc)
                        cond_states[ci] = cond_state + _r(c_g_sa).to(residual_dtype) * cond_attn.to(residual_dtype)

                    noise_attn = rearrange(noise_std, "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                    x_5d = x_5d + _r(g_sa).to(residual_dtype) * noise_attn.to(residual_dtype)

            # Cross-attention
            norm_ca = block.layer_norm_cross_attn(x_5d) * (1 + _r(sc_ca)) + _r(sh_ca)
            ca_out = rearrange(
                block.cross_attn(
                    rearrange(norm_ca.to(compute_dtype), "b t h w d -> b (t h w) d"),
                    context, rope_emb=block_kwargs["rope_emb_L_1_1_D"],
                    transformer_options=transformer_options,
                ),
                "b (t h w) d -> b t h w d", t=T, h=H, w=W,
            )
            x_5d = x_5d + _r(g_ca).to(residual_dtype) * ca_out.to(residual_dtype)

            # MLP
            norm_ml = block.layer_norm_mlp(x_5d) * (1 + _r(sc_ml)) + _r(sh_ml)
            x_5d = x_5d + _r(g_ml).to(residual_dtype) * block.mlp(norm_ml.to(compute_dtype)).to(residual_dtype)

            if not use_cache and any_active:
                for ci, cond_state in enumerate(cond_states):
                    s, e = cond_block_ranges[ci]
                    if not (s <= bi <= e):
                        continue
                    if block.use_adaln_lora:
                        c_sh_ml, c_sc_ml, c_g_ml = (block.adaln_modulation_mlp(cond_t_emb) + cond_adaln_lora).chunk(3, -1)
                    else:
                        c_sh_ml, c_sc_ml, c_g_ml = block.adaln_modulation_mlp(cond_t_emb).chunk(3, -1)
                    c_norm_ml = block.layer_norm_mlp(cond_state) * (1 + _r(c_sc_ml)) + _r(c_sh_ml)
                    cond_states[ci] = cond_state + _r(c_g_ml).to(residual_dtype) * block.mlp(c_norm_ml.to(compute_dtype)).to(residual_dtype)

        x_out = dit.final_layer(x_5d.to(context.dtype), t_emb, adaln_lora_B_T_3D=adaln_lora)
        result = dit.unpatchify(x_out)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]
        return result

    comfy.patcher_extension.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "easycontrol_anima_exp",
        easycontrol_exp_wrapper,
        model_patcher.model_options,
        is_model_options=True,
    )


# ============================================================================
# Anima Reference Latent — temporal concatenation
# ============================================================================

class AnimaReferenceLatent:
    """Temporal concat control for Anima.

    Concatenates a reference image (VAE-encoded) as temporal frame T=1
    alongside the noise frame T=0. The model's 3D RoPE naturally encodes
    the position difference. Works with standard PEFT LoRAs loaded via
    ComfyUI's built-in Load LoRA node.

    Workflow: Load Checkpoint → Load LoRA → AnimaReferenceLatent → KSampler
    No custom LoRA loader needed — uses ComfyUI's standard LoRA Loader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ref_latent": ("LATENT", {"tooltip": "VAE-encoded reference image (use VAE Encode)"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "Start applying at this % of sampling (0.0 = from start)"}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Stop applying at this % of sampling (1.0 = until end)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima"

    def apply(self, model, ref_latent, start_percent, end_percent):
        model_clone = model.clone()

        ref_samples = ref_latent["samples"]
        if ref_samples.ndim == 4:
            ref_samples = ref_samples.unsqueeze(2)

        ref_samples = model_clone.model.process_latent_in(ref_samples)

        model_clone.model_options["anima_ref_latents"] = ref_samples
        model_clone.model_options["anima_ref_start_percent"] = start_percent
        model_clone.model_options["anima_ref_end_percent"] = end_percent

        # Register the DIFFUSION_MODEL wrapper
        _register_ref_latent_wrapper(model_clone)

        return (model_clone,)


def _register_ref_latent_wrapper(model_patcher):
    """Wrapper: concatenate reference latent as temporal frame before _forward."""

    _options_ref = model_patcher.model_options

    def ref_latent_wrapper(executor, *args, **kwargs):
        ref_latents = _options_ref.get("anima_ref_latents", None)

        if ref_latents is None:
            return executor(*args, **kwargs)

        # Track progress using sample_sigmas (reliable across runs)
        current_t = args[1].max().item()
        sample_sigmas = kwargs.get("transformer_options", {}).get("sample_sigmas", None)
        if sample_sigmas is not None:
            max_sigma = sample_sigmas.max().item()
            progress = 1.0 - (current_t / max_sigma) if max_sigma > 0 else 0.0
        else:
            progress = 0.0

        start_pct = _options_ref.get("anima_ref_start_percent", 0.0)
        end_pct = _options_ref.get("anima_ref_end_percent", 1.0)

        if not (start_pct <= progress <= end_pct):
            # Outside active range — continue chain without concat
            return executor(*args, **kwargs)

        # x is (B, C, T, H, W) — the noisy latent
        x = args[0]
        rest_args = args[1:]

        # Move ref to same device/dtype as x
        ref = ref_latents.to(device=x.device, dtype=x.dtype)

        # Expand ref to match batch size (CFG batches cond+uncond together)
        if ref.shape[0] < x.shape[0]:
            ref = ref.expand(x.shape[0], -1, -1, -1, -1)

        # Resize ref spatial dims to match noise if different
        if ref.shape[3] != x.shape[3] or ref.shape[4] != x.shape[4]:
            ref = torch.nn.functional.interpolate(
                ref.squeeze(2),
                size=(x.shape[3], x.shape[4]),
                mode='bilinear', align_corners=False,
            ).unsqueeze(2)

        # Temporal concat: [noise_T0, reference_T1]
        x_concat = torch.cat([x, ref], dim=2)
        orig_T = x.shape[2]

        # Call _forward directly with concatenated input
        output = executor.class_obj._forward(x_concat, *rest_args, **kwargs)

        if output.shape[2] > orig_T:
            output = output[:, :, :orig_T, :, :]

        return output

    comfy.patcher_extension.add_wrapper_with_key(
        WrappersMP.DIFFUSION_MODEL,
        "anima_ref_latent",
        ref_latent_wrapper,
        model_patcher.model_options,
        is_model_options=True,
    )


# ============================================================================
# Node registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "LoadEasyControl": LoadEasyControl,
    "ApplyEasyControlCondition": ApplyEasyControlCondition,
    "ApplyEasyControlConditionPlus": ApplyEasyControlConditionPlus,
    "ApplyEasyControlExperimental": ApplyEasyControlExperimental,
    "AnimaReferenceLatent": AnimaReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadEasyControl": "Load EasyControl (Anima)",
    "ApplyEasyControlCondition": "Apply EasyControl Condition (Anima)",
    "ApplyEasyControlConditionPlus": "Apply EasyControl Condition+ (Anima) [LoRA-safe]",
    "ApplyEasyControlExperimental": "Apply EasyControl (Anima) - Experimental",
    "AnimaReferenceLatent": "Anima Reference Latent (Temporal Concat)",
}
