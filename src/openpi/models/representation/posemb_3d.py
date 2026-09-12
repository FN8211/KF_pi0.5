"""
3D posemb_sincos, Adapted from Qwen-2.5VL M-RoPE

We assume the raw image patches are 16x16.

Spatial index for 4x4 patches (pooling from 16x16 to 4x4):
      2           6           10         14
.-----------.-----------.-----------.-----------.
|           |           |           |           |
.-----------.-----------.-----------.-----------.

Temporal index: relative distance from current step (current - past).

View index: 0 for base camera, 1 for wrist camera.

dim is split into 8 equal parts:
  temporal:    sin + cos = 2 parts
  view:        sin + cos = 2 parts
  spatial_row: sin + cos = 2 parts
  spatial_col: sin + cos = 2 parts
"""

import jax.numpy as jnp
import einops


class PosEmb3D:
    def __init__(self, dim: int, temporal_base: int = 1000, spatial_base: int = 1_000, view_base: int = 100):
        assert dim % 8 == 0, "dim must be divisible by 8"
        self.dim = dim
        width = dim // 8  # each of 4 components gets 2*width dims (sin+cos)

        omega_t = jnp.arange(width) / max(width - 1, 1)
        omega_s = jnp.arange(width) / max(width - 1, 1)
        omega_v = jnp.arange(width) / max(width - 1, 1)

        self.temporal_omega = 1.0 / (temporal_base ** omega_t)
        self.spatial_omega  = 1.0 / (spatial_base  ** omega_s)
        self.view_omega     = 1.0 / (view_base     ** omega_v)

        # Pre-compute spatial embeddings for supported pool sizes
        self.spatial_pe8x8 = self._compute_spatial_pe(8, stride=2)
        self.spatial_pe4x4 = self._compute_spatial_pe(4, stride=4)
        self.spatial_pe2x2 = self._compute_spatial_pe(2, stride=8)

        # Pre-compute view embeddings for V=2
        self.view_pe = self._compute_view_pe(max_views=8)

        # Pre-compute temporal PE cache
        self.temporal_pe = self._compute_temporal_pe(max_length=1000)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _compute_spatial_pe(self, grid_size: int, stride: int):
        """Compute spatial PE for a grid_size x grid_size patch grid.
        stride: distance between adjacent patch centers in original 16x16 space.
        """
        y, x = jnp.mgrid[:grid_size, :grid_size]
        y = stride * y + stride // 2  # center of each patch
        x = stride * x + stride // 2

        y_sin = jnp.einsum("m,d->md", y.flatten().astype(float), self.spatial_omega)
        x_sin = jnp.einsum("m,d->md", x.flatten().astype(float), self.spatial_omega)

        return jnp.concatenate(
            [jnp.sin(y_sin), jnp.cos(y_sin), jnp.sin(x_sin), jnp.cos(x_sin)], axis=-1
        )  # [grid_size*grid_size, 4*width]

    def _compute_view_pe(self, max_views: int):
        """Compute view PE for up to max_views views."""
        pos = jnp.arange(max_views).astype(float)
        sinusoid = jnp.einsum("m,d->md", pos, self.view_omega)
        return jnp.concatenate(
            [jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1
        )  # [max_views, 2*width]

    def _compute_temporal_pe(self, max_length: int):
        """Compute temporal PE table of length max_length."""
        pos = jnp.arange(max_length).astype(float)
        sinusoid = jnp.einsum("m,d->md", pos, self.temporal_omega)
        return jnp.concatenate(
            [jnp.sin(sinusoid), jnp.cos(sinusoid)], axis=-1
        )  # [max_length, 2*width]

    # ── public API ────────────────────────────────────────────────────────────

    def __call__(
        self,
        relative_pos,   # [B, T]  relative temporal distance (current_step - past_step)
        num_views: int, # V, number of camera views
        spatial_size: int = 4,  # pool target grid size (4 → 4x4=16 patches)
    ):
        """
        Generate 4D positional embeddings (temporal + view + spatial).

        Args:
            relative_pos:  [B, T] int array, relative temporal distances
            num_views:     number of views V
            spatial_size:  spatial grid size after pooling (4 → 16 patches)

        Returns:
            pe: [B, T, V, spatial_size*spatial_size, dim]
        """
        B, T = relative_pos.shape
        HW = spatial_size * spatial_size

        # ── temporal PE ──────────────────────────────────────────────────────
        t_pe = self.temporal_pe[relative_pos]           # [B, T, 2*width]
        t_pe = einops.repeat(t_pe, "b t d -> b t v hw d", v=num_views, hw=HW)

        # ── view PE ───────────────────────────────────────────────────────────
        view_ids = jnp.arange(num_views)
        v_pe = self.view_pe[view_ids]                   # [V, 2*width]
        v_pe = einops.repeat(v_pe, "v d -> b t v hw d", b=B, t=T, hw=HW)

        # ── spatial PE ────────────────────────────────────────────────────────
        assert spatial_size in [2, 4, 8], "spatial_size must be 2, 4, or 8"
        if spatial_size == 8:
            s_pe = self.spatial_pe8x8
        elif spatial_size == 4:
            s_pe = self.spatial_pe4x4
        else:
            s_pe = self.spatial_pe2x2

        s_pe = einops.repeat(s_pe, "hw d -> b t v hw d", b=B, t=T, v=num_views)

        # ── concatenate all ───────────────────────────────────────────────────
        pe = jnp.concatenate([t_pe, v_pe, s_pe], axis=-1)  # [B, T, V, HW, dim]
        return pe


if __name__ == "__main__":
    posemb = PosEmb3D(dim=2048)
    B, T, V = 2, 6, 2
    relative_pos = jnp.array([[0, 10, 50, 100, 200, 300]] * B)
    pe = posemb(relative_pos, num_views=V, spatial_size=4)
    print("pe shape:", pe.shape)  # should be [2, 6, 2, 16, 2048]