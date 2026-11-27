"""Target-only graph fusion used by the PCFB backbone."""


import os
from utils.experiment_options import experiment_option
import torch

from models.pcfb import Model as PCFB


class Model(PCFB):
    def __init__(self, configs, d_inputs, pe="zeros", learn_pe=True):
        # Default settings for this dedicated experiment file.
        # They can still be overridden by environment variables in the run script.
        os.environ.setdefault("PCFB_FUSION_MODE", experiment_option("PCFB_FUSION_MODE", "input"))
        os.environ.setdefault("PCFB_TARGET_ONLY_FUSE", experiment_option("PCFB_TARGET_ONLY_FUSE", "1"))
        os.environ.setdefault("PCFB_LAG", experiment_option("PCFB_LAG", "1"))
        os.environ.setdefault("PCFB_BASE_SCALE", experiment_option("PCFB_BASE_SCALE", "0.05"))
        os.environ.setdefault("PCFB_RESIDUAL_SCALE", experiment_option("PCFB_RESIDUAL_SCALE", "0.03"))
        os.environ.setdefault("PCFB_STRENGTH", experiment_option("PCFB_STRENGTH", "0.01"))
        os.environ.setdefault("PCFB_ALPHA_INIT", experiment_option("PCFB_ALPHA_INIT", "0.001"))

        super().__init__(
            configs=configs,
            d_inputs=d_inputs,
            pe=pe,
            learn_pe=learn_pe,
        )

        if self.uncle_lag != 1:
            raise ValueError(
                "PCFB graph fusion currently requires PCFB_LAG=1"
            )

        self.uncle_base_scale = float(experiment_option("PCFB_BASE_SCALE", "0.05"))
        self.uncle_residual_scale = float(experiment_option("PCFB_RESIDUAL_SCALE", "0.03"))

        if self.uncle_base_scale < 0:
            raise ValueError("PCFB_BASE_SCALE must be non-negative")
        if self.uncle_residual_scale < 0:
            raise ValueError("PCFB_RESIDUAL_SCALE must be non-negative")

        # Start from the fixed relationship graph.
        # The learnable residual is initialized to zero.
        with torch.no_grad():
            self.var_mat.zero_()

        self.var_mat.requires_grad_(True)

        print("[PCFB graph fusion] initialized")
        print("[PCFB graph fusion] fusion_mode:", self.uncle_fusion_mode)
        print("[PCFB graph fusion] target_only_fuse:", self.uncle_target_only_fuse)
        print("[PCFB graph fusion] target_index:", self.uncle_target_index)
        print("[PCFB graph fusion] base_scale:", self.uncle_base_scale)
        print("[PCFB graph fusion] residual_scale:", self.uncle_residual_scale)
        print("[PCFB graph fusion] residual init: zero")
        print("[PCFB graph fusion] var_mat trainable:", self.var_mat.requires_grad)


    def _is_zero_bypass(self):
        return (
            abs(float(getattr(self, "uncle_base_scale", 0.0))) < 1e-12
            and abs(float(getattr(self, "uncle_residual_scale", 0.0))) < 1e-12
        )

    def forward(self, x, return_rep=False):
        """Strict Alpha-equivalent path when graph injection is disabled.

        This bypass is used only for zero-control experiments:
        PCFB_BASE_SCALE=0 and PCFB_RESIDUAL_SCALE=0.
        """
        if self._is_zero_bypass():
            if self.rev_in:
                x = self.revin_layer(x, 'norm')

            result = self.decoder(x, return_rep=return_rep)

            if return_rep:
                ys, reps = result
            else:
                ys = result

            if self.rev_in:
                ys = [self.revin_layer(y, 'denorm') for y in ys]

            if return_rep:
                return ys, reps
            else:
                return ys

        return super().forward(x, return_rep=return_rep)

    def _effective_var_mat(self):
        """Combine fixed relationship graph and learnable bounded residual."""

        dtype = self.var_mat.dtype
        device = self.var_mat.device

        gate = self.prior_gate.to(device=device, dtype=dtype).unsqueeze(-1)
        off_diag = self.off_diag.to(device=device, dtype=dtype).unsqueeze(-1)

        base_relation = self.uncle_base_scale * gate
        residual_relation = self.uncle_residual_scale * torch.tanh(self.var_mat)

        effective_var_mat = (base_relation + residual_relation) * off_diag

        # Keep only edges pointing to the target variable when
        # PCFB_TARGET_ONLY_FUSE=1.
        effective_var_mat = self._maybe_keep_target_column_only(effective_var_mat)

        return effective_var_mat
