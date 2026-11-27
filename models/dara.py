"""Dynamic Adaptive Residual Adapter (DARA).

The paper configuration uses the local and zero-correction branches.
"""


import os
from utils.experiment_options import experiment_option

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.relation import Model as GraphFusedPCFB


def _env_flag(name, default="0"):
    return experiment_option(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class _TemporalResidualHead(nn.Module):
    """Map a fixed-length history vector to one target residual horizon."""

    def __init__(self, input_size, hidden_size, pred_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, pred_len),
        )

    def forward(self, history):
        return self.net(history).unsqueeze(-1)


class Model(GraphFusedPCFB):
    uses_input_observation_mask = True

    def __init__(
        self,
        configs,
        d_inputs,
        pe="zeros",
        learn_pe=True,
    ):
        # V2 supplies graph loading and an Alpha-equivalent standard decoder.
        # Its early input-fusion path is not used by DARA.
        os.environ["PCFB_FUSION_MODE"] = "input"
        os.environ["PCFB_TARGET_ONLY_FUSE"] = "1"
        os.environ["PCFB_LAG"] = "1"
        os.environ["PCFB_RESIDUAL_SCALE"] = "0.0"
        os.environ.setdefault("PCFB_BASE_SCALE", experiment_option("PCFB_BASE_SCALE", "1.0"))

        super().__init__(
            configs=configs,
            d_inputs=d_inputs,
            pe=pe,
            learn_pe=learn_pe,
        )

        self.input_len = int(d_inputs[0])
        self.n_vars = int(d_inputs[-1])
        self.pred_len = int(configs.pred_len)

        self.dara_disable = _env_flag("DARA_DISABLE")
        self.base_causal_fusion = _env_flag(
            "DARA_BASE_CAUSAL_FUSION"
        )
        self.disable_causal = _env_flag("DARA_DISABLE_CAUSAL")
        self.disable_trend = _env_flag("DARA_DISABLE_TREND")
        self.disable_local = _env_flag("DARA_DISABLE_LOCAL")

        self.var_mat.requires_grad_(False)
        self.alpha_logit.requires_grad_(self.base_causal_fusion)

        self.expert_hidden = int(
            experiment_option("DARA_EXPERT_HIDDEN", "32")
        )
        self.router_hidden = int(
            experiment_option("DARA_ROUTER_HIDDEN", "16")
        )
        self.local_context = min(
            self.input_len,
            int(experiment_option("DARA_LOCAL_CONTEXT", "48")),
        )
        self.trend_kernel = int(
            experiment_option("DARA_TREND_KERNEL", "25")
        )
        self.null_bias = float(
            experiment_option("DARA_NULL_BIAS", "1.0")
        )
        self.max_late_scale = float(
            experiment_option("DARA_MAX_LATE_SCALE", "0.25")
        )
        self.correction_clip = float(
            experiment_option("DARA_CORRECTION_CLIP", "0.5")
        )
        self.history_min_parent_coverage = float(
            experiment_option(
                "DARA_HISTORY_MIN_PARENT_COVERAGE",
                "0.1",
            )
        )
        self.eps = float(experiment_option("DARA_EPS", "1e-6"))

        if self.expert_hidden < 1 or self.router_hidden < 1:
            raise ValueError("DARA hidden dimensions must be positive")
        if self.local_context < 2:
            raise ValueError("DARA_LOCAL_CONTEXT must be at least 2")
        if self.trend_kernel < 1:
            raise ValueError("DARA_TREND_KERNEL must be positive")
        if self.trend_kernel % 2 == 0:
            raise ValueError("DARA_TREND_KERNEL must be odd")
        if self.max_late_scale < 0.0:
            raise ValueError("DARA_MAX_LATE_SCALE must be non-negative")
        if self.correction_clip <= 0.0:
            raise ValueError("DARA_CORRECTION_CLIP must be positive")
        if not 0.0 <= self.history_min_parent_coverage <= 1.0:
            raise ValueError(
                "DARA_HISTORY_MIN_PARENT_COVERAGE must be in [0, 1]"
            )

        # Preserve the random trajectory of the Alpha decoder. The additional
        # modules use no dropout, so restoring this state also keeps strict
        # disabled-control runs aligned batch by batch.
        rng_state_cpu = torch.get_rng_state()
        rng_state_cuda = None
        if torch.cuda.is_available():
            try:
                rng_state_cuda = torch.cuda.get_rng_state_all()
            except Exception:
                rng_state_cuda = None

        self.causal_head = _TemporalResidualHead(
            self.input_len,
            self.expert_hidden,
            self.pred_len,
        )
        self.trend_head = _TemporalResidualHead(
            self.input_len,
            self.expert_hidden,
            self.pred_len,
        )
        self.local_head = _TemporalResidualHead(
            2 * self.local_context,
            self.expert_hidden,
            self.pred_len,
        )

        self.router_feature_dim = 9
        self.branch_router = nn.Sequential(
            nn.Linear(self.router_feature_dim, self.router_hidden),
            nn.GELU(),
            nn.Linear(
                self.router_hidden,
                self.pred_len * 4,
            ),
        )
        self.sample_gate_router = nn.Sequential(
            nn.Linear(self.router_feature_dim, self.router_hidden),
            nn.GELU(),
            nn.Linear(self.router_hidden, self.pred_len),
        )

        # Start with horizon-specific but sample-neutral routing. The null
        # expert receives a modest prior, while all corrections remain exactly
        # zero because late_scale_raw starts at zero.
        branch_out = self.branch_router[-1]
        nn.init.zeros_(branch_out.weight)
        nn.init.zeros_(branch_out.bias)
        with torch.no_grad():
            branch_out.bias.view(self.pred_len, 4)[:, 0] = self.null_bias

        gate_out = self.sample_gate_router[-1]
        nn.init.zeros_(gate_out.weight)
        nn.init.zeros_(gate_out.bias)

        self.late_scale_raw = nn.Parameter(torch.zeros(()))
        self.causal_history_gain = nn.Parameter(torch.ones(()))
        self.causal_history_bias = nn.Parameter(torch.zeros(()))

        target_selector = torch.zeros(self.n_vars)
        target_selector[self.uncle_target_index] = 1.0
        self.register_buffer("target_selector", target_selector)

        torch.set_rng_state(rng_state_cpu)
        if rng_state_cuda is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng_state_cuda)
            except Exception:
                pass

        # Diagnostics: null, causal, trend, local, late scale, sample gate,
        # correction RMS, parent coverage.
        self.register_buffer(
            "_chsrf_diag_sum",
            torch.zeros(8, dtype=torch.float64),
        )
        self.register_buffer(
            "_chsrf_diag_count",
            torch.zeros((), dtype=torch.float64),
        )

        print("[DARA] dynamic residual adaptation")
        print("[DARA] target_index:", self.uncle_target_index)
        print(
            "[DARA] base_causal_fusion:",
            self.base_causal_fusion,
        )
        print("[DARA] expert_hidden:", self.expert_hidden)
        print("[DARA] router_hidden:", self.router_hidden)
        print("[DARA] local_context:", self.local_context)
        print("[DARA] trend_kernel:", self.trend_kernel)
        print("[DARA] null_bias:", self.null_bias)
        print("[DARA] max_late_scale:", self.max_late_scale)
        print("[DARA] correction_clip:", self.correction_clip)
        print(
            "[DARA] enabled branches:",
            {
                "causal": not self.disable_causal,
                "trend": not self.disable_trend,
                "local": not self.disable_local,
            },
        )
        print("[DARA] disabled:", self.dara_disable)

    def _prepare_input_mask(self, x, input_mask):
        if input_mask is None:
            return torch.ones_like(x)
        if input_mask.shape != x.shape:
            raise ValueError(
                "input_mask and x must have identical shapes, "
                f"received {tuple(input_mask.shape)} and {tuple(x.shape)}"
            )
        return (
            torch.nan_to_num(
                input_mask.to(device=x.device, dtype=x.dtype),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
            .clamp(0.0, 1.0)
            .detach()
        )

    def _parent_weights(self, x):
        weights = (
            self.prior_gate[:, self.uncle_target_index]
            .to(device=x.device, dtype=x.dtype)
            .clamp_min(0.0)
            .clone()
        )
        weights[self.uncle_target_index] = 0.0
        total = weights.sum()
        if float(total.detach().cpu()) <= self.eps:
            raise RuntimeError(
                "DARA causal expert has no positive parent edge "
                "pointing to the target"
            )
        return weights / total.clamp_min(self.eps)

    def _parent_reference(self, x, mask):
        weights = self._parent_weights(x)
        weighted_mask = mask * weights.view(1, 1, -1)
        observed_mass = weighted_mask.sum(dim=-1)
        observed_reference = (
            (x * weighted_mask).sum(dim=-1)
            / observed_mass.clamp_min(self.eps)
        )
        fallback_reference = (
            x * weights.view(1, 1, -1)
        ).sum(dim=-1)
        reference = torch.where(
            observed_mass > self.eps,
            observed_reference,
            fallback_reference,
        )
        return reference, observed_mass.clamp(0.0, 1.0)

    def _moving_average(self, target):
        kernel = min(self.trend_kernel, target.size(1))
        if kernel % 2 == 0:
            kernel = max(1, kernel - 1)
        if kernel == 1:
            return target
        pad = kernel // 2
        padded = F.pad(
            target.unsqueeze(1),
            (pad, pad),
            mode="replicate",
        )
        return F.avg_pool1d(
            padded,
            kernel_size=kernel,
            stride=1,
        ).squeeze(1)

    def _router_features(
        self,
        target,
        trend,
        mask,
        parent_coverage,
    ):
        recent_len = self.local_context
        recent = target[:, -recent_len:]
        recent_trend = trend[:, -recent_len:]
        local_component = recent - recent_trend

        target_energy = target.float().square().mean(
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)
        trend_energy = trend.float().square().mean(
            dim=1,
            keepdim=True,
        )
        local_energy = local_component.float().square().mean(
            dim=1,
            keepdim=True,
        )

        target_mask = mask[
            :,
            :,
            self.uncle_target_index:
            self.uncle_target_index + 1,
        ]

        features = torch.cat(
            [
                target.mean(dim=1, keepdim=True),
                target.std(
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                ),
                recent.mean(dim=1, keepdim=True),
                recent.std(
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                ),
                recent[:, -1:] - recent[:, :1],
                trend_energy / target_energy,
                local_energy / target_energy,
                target_mask.mean(dim=1),
                parent_coverage.mean(dim=1, keepdim=True),
            ],
            dim=1,
        )
        return torch.nan_to_num(
            features,
            nan=0.0,
            posinf=5.0,
            neginf=-5.0,
        ).clamp(-5.0, 5.0)

    def _branch_residuals(self, x, mask):
        target = x[..., self.uncle_target_index]
        parent_reference, parent_coverage = self._parent_reference(
            x,
            mask,
        )

        causal_reference = (
            self.causal_history_gain * parent_reference
            + self.causal_history_bias
        )
        causal_history = causal_reference - target
        trend = self._moving_average(target)

        recent_target = target[:, -self.local_context:]
        target_diff = F.pad(
            target[:, 1:] - target[:, :-1],
            (1, 0),
        )
        recent_diff = target_diff[:, -self.local_context:]
        local_history = torch.cat(
            [recent_target, recent_diff],
            dim=1,
        )

        causal_residual = self.causal_head(causal_history)
        trend_residual = self.trend_head(trend)
        local_residual = self.local_head(local_history)

        router_features = self._router_features(
            target,
            trend,
            mask,
            parent_coverage,
        )
        route_logits = self.branch_router(
            router_features
        ).view(-1, self.pred_len, 4)

        disabled = torch.tensor(
            [
                False,
                self.disable_causal,
                self.disable_trend,
                self.disable_local,
            ],
            device=route_logits.device,
            dtype=torch.bool,
        )
        route_logits = route_logits.masked_fill(
            disabled.view(1, 1, 4),
            -1e4,
        )
        branch_weights = torch.softmax(route_logits, dim=-1)

        branch_values = torch.stack(
            [
                torch.zeros_like(causal_residual),
                causal_residual,
                trend_residual,
                local_residual,
            ],
            dim=-1,
        )
        fused = (
            branch_values
            * branch_weights.unsqueeze(2)
        ).sum(dim=-1)

        sample_gate = torch.sigmoid(
            self.sample_gate_router(router_features)
        ).unsqueeze(-1)
        late_scale = (
            self.max_late_scale
            * torch.tanh(self.late_scale_raw)
        )
        bounded_fused = (
            self.correction_clip
            * torch.tanh(
                fused / self.correction_clip
            )
        )
        correction = late_scale * sample_gate * bounded_fused

        self._record_dara_diagnostics(
            branch_weights,
            late_scale,
            sample_gate,
            correction,
            parent_coverage,
        )
        return correction

    def compute_training_auxiliary_loss(
        self,
        x,
        input_mask=None,
    ):
        """Predict observed target history from lagged graph parents."""
        if self.dara_disable or self.disable_causal:
            return x.sum() * 0.0

        mask = self._prepare_input_mask(x, input_mask)
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        ).detach()
        x_norm = (x - mean) / (std + self.eps)

        parent_reference, parent_coverage = self._parent_reference(
            x_norm,
            mask,
        )
        pred_target = (
            self.causal_history_gain
            * parent_reference[:, :-1]
            + self.causal_history_bias
        )
        true_target = x_norm[
            :,
            1:,
            self.uncle_target_index,
        ]
        target_mask = mask[
            :,
            1:,
            self.uncle_target_index,
        ]
        valid_parent = (
            parent_coverage[:, :-1]
            >= self.history_min_parent_coverage
        ).to(dtype=x.dtype)
        valid = target_mask * valid_parent

        loss = F.smooth_l1_loss(
            pred_target,
            true_target,
            reduction="none",
        )
        observed = valid.sum()
        if float(observed.detach().cpu()) <= 0.0:
            return (
                self.causal_history_gain
                + self.causal_history_bias
            ) * 0.0
        return (loss * valid).sum() / observed

    @torch.no_grad()
    def _record_dara_diagnostics(
        self,
        branch_weights,
        late_scale,
        sample_gate,
        correction,
        parent_coverage,
    ):
        batch_size = branch_weights.size(0)
        mean_weights = branch_weights.float().mean(dim=(0, 1))
        correction_rms = torch.sqrt(
            correction.float().square().mean().clamp_min(0.0)
        )
        values = torch.cat(
            [
                mean_weights,
                late_scale.float().reshape(1),
                sample_gate.float().mean().reshape(1),
                correction_rms.reshape(1),
                parent_coverage.float().mean().reshape(1),
            ]
        ).to(self._chsrf_diag_sum.device, dtype=torch.float64)
        self._chsrf_diag_sum.add_(values * batch_size)
        self._chsrf_diag_count.add_(batch_size)

    @torch.no_grad()
    def reset_dara_diagnostics(self):
        self._chsrf_diag_sum.zero_()
        self._chsrf_diag_count.zero_()

    @torch.no_grad()
    def dara_diagnostics(self):
        count = float(self._chsrf_diag_count.item())
        if count <= 0.0:
            return {}
        values = self._chsrf_diag_sum / count
        return {
            "null_weight_mean": float(values[0].cpu()),
            "causal_weight_mean": float(values[1].cpu()),
            "trend_weight_mean": float(values[2].cpu()),
            "local_weight_mean": float(values[3].cpu()),
            "late_scale": float(values[4].cpu()),
            "sample_gate_mean": float(values[5].cpu()),
            "correction_rms": float(values[6].cpu()),
            "parent_coverage_mean": float(values[7].cpu()),
        }

    def forward(
        self,
        x,
        input_mask=None,
        return_rep=False,
    ):
        mask = self._prepare_input_mask(x, input_mask)

        if self.rev_in:
            x = self.revin_layer(x, "norm")

        expert_x = x
        if self.base_causal_fusion:
            effective_var_mat = self._effective_var_mat().to(
                device=x.device,
                dtype=x.dtype,
            )
            alpha = self._alpha().to(
                device=x.device,
                dtype=x.dtype,
            )
            causal_message = self._causal_message(
                x,
                effective_var_mat,
            )
            if alpha.ndim == 0:
                x = x + alpha * causal_message
            else:
                x = (
                    x
                    + alpha.view(1, 1, -1)
                    * causal_message
                )

        result = self.decoder(
            x,
            return_rep=return_rep,
        )
        if return_rep:
            base_outputs, reps = result
        else:
            base_outputs = result

        if self.dara_disable:
            outputs = base_outputs
        else:
            correction = self._branch_residuals(
                expert_x,
                mask,
            )
            selector = self.target_selector.to(
                device=x.device,
                dtype=x.dtype,
            ).view(1, 1, -1)
            outputs = [
                output
                + correction[:, :output.size(1), :]
                * selector
                for output in base_outputs
            ]

        if self.rev_in:
            outputs = [
                self.revin_layer(output, "denorm")
                for output in outputs
            ]

        if return_rep:
            return outputs, reps
        return outputs
