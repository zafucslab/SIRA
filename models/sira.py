"""SIRA forecasting model with a sparse multi-lag residual."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dara import Model as DARA


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _parse_lags(value):
    lags = sorted({int(item.strip()) for item in value.split(",")})
    if not lags or lags[0] < 1:
        raise ValueError("SGR_LAGS must contain positive integers")
    return lags


class Model(DARA):
    """PCFB + local DARA + Sparse Granger Residual (SGR)."""

    uses_input_observation_mask = True

    def __init__(
        self,
        configs,
        d_inputs,
        pe="zeros",
        learn_pe=True,
    ):
        super().__init__(
            configs=configs,
            d_inputs=d_inputs,
            pe=pe,
            learn_pe=learn_pe,
        )

        self.sgr_disable = _env_flag("SGR_DISABLE", "1")
        self.sgr_lags = _parse_lags(
            os.environ.get("SGR_LAGS", "1,3,7,14,28")
        )
        self.sgr_hidden_size = int(
            os.environ.get("SGR_HIDDEN_SIZE", "32")
        )
        self.sgr_null_bias = float(
            os.environ.get("SGR_NULL_BIAS", "2.0")
        )
        self.sgr_max_scale = float(
            os.environ.get("SGR_MAX_SCALE", "0.10")
        )
        self.sgr_correction_clip = float(
            os.environ.get("SGR_CORRECTION_CLIP", "0.5")
        )
        self.sgr_var_lambda = float(
            os.environ.get("SGR_VAR_LAMBDA", "0.01")
        )
        self.sgr_lag_lambda = float(
            os.environ.get("SGR_LAG_LAMBDA", "0.005")
        )
        self.sgr_self_loss_weight = float(
            os.environ.get("SGR_SELF_LOSS_WEIGHT", "0.25")
        )

        if self.sgr_lags[-1] >= self.input_len:
            raise ValueError(
                "largest SGR lag must be smaller than the input length"
            )
        if self.sgr_hidden_size < 1:
            raise ValueError("SGR_HIDDEN_SIZE must be positive")
        if self.sgr_max_scale < 0.0:
            raise ValueError("SGR_MAX_SCALE must be non-negative")
        if self.sgr_correction_clip <= 0.0:
            raise ValueError("SGR_CORRECTION_CLIP must be positive")
        if self.sgr_var_lambda < 0.0 or self.sgr_lag_lambda < 0.0:
            raise ValueError("SGR sparsity weights must be non-negative")
        if self.sgr_self_loss_weight < 0.0:
            raise ValueError("SGR_SELF_LOSS_WEIGHT must be non-negative")

        rng_state_cpu = torch.get_rng_state()
        rng_state_cuda = None
        if torch.cuda.is_available():
            try:
                rng_state_cuda = torch.cuda.get_rng_state_all()
            except Exception:
                rng_state_cuda = None

        hidden_size = self.sgr_hidden_size
        n_lags = len(self.sgr_lags)

        # Value and observation-mask channels share one grouped support.
        self.sgr_self_weight = nn.Parameter(
            torch.empty(hidden_size, n_lags, 2)
        )
        self.sgr_parent_weight = nn.Parameter(
            torch.empty(hidden_size, self.n_vars, n_lags, 2)
        )
        self.sgr_self_bias = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.sgr_self_weight, mean=0.0, std=0.02)
        nn.init.normal_(self.sgr_parent_weight, mean=0.0, std=0.02)

        parent_selector = torch.ones(self.n_vars)
        parent_selector[self.uncle_target_index] = 0.0
        self.register_buffer("sgr_parent_selector", parent_selector)
        with torch.no_grad():
            self.sgr_parent_weight[:, self.uncle_target_index].zero_()

        self.sgr_self_history_head = nn.Linear(hidden_size, 1)
        self.sgr_parent_history_head = nn.Linear(
            hidden_size,
            1,
            bias=False,
        )
        nn.init.zeros_(self.sgr_parent_history_head.weight)

        self.sgr_condition = nn.Linear(hidden_size, hidden_size)
        self.sgr_residual_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, self.pred_len, bias=False),
        )
        self.sgr_gate_head = nn.Sequential(
            nn.Linear(2 * hidden_size + 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.pred_len),
        )
        nn.init.zeros_(self.sgr_gate_head[-1].weight)
        nn.init.constant_(
            self.sgr_gate_head[-1].bias,
            -self.sgr_null_bias,
        )

        # The full model is exactly equal to its control at initialization.
        self.sgr_scale_raw = nn.Parameter(torch.zeros(()))

        # Plain attributes are intentionally excluded from checkpoints. They
        # are used only for validation-time block permutation audits.
        self.sgr_audit_variable = -1
        self.sgr_audit_block_size = 7
        self.sgr_runtime_parent_indices = None

        torch.set_rng_state(rng_state_cpu)
        if rng_state_cuda is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(rng_state_cuda)
            except Exception:
                pass

        # scale, gate, correction RMS, parent coverage, target coverage
        self.register_buffer(
            "_sgr_diag_sum",
            torch.zeros(5, dtype=torch.float64),
        )
        self.register_buffer(
            "_sgr_diag_count",
            torch.zeros((), dtype=torch.float64),
        )

        print("[SGR] Sparse Granger Residual (SGR)")
        print("[SGR] disabled:", self.sgr_disable)
        print("[SGR] target_index:", self.uncle_target_index)
        print("[SGR] lags:", self.sgr_lags)
        print("[SGR] hidden_size:", self.sgr_hidden_size)
        print("[SGR] null_bias:", self.sgr_null_bias)
        print("[SGR] max_scale:", self.sgr_max_scale)
        print("[SGR] variable_group_lambda:", self.sgr_var_lambda)
        print("[SGR] lag_group_lambda:", self.sgr_lag_lambda)

    def _effective_parent_weight(self):
        selector = self.sgr_parent_selector.to(
            device=self.sgr_parent_weight.device,
            dtype=self.sgr_parent_weight.dtype,
        )
        if self.sgr_runtime_parent_indices is not None:
            runtime_selector = torch.zeros_like(selector)
            for variable_index in self.sgr_runtime_parent_indices:
                runtime_selector[variable_index] = 1.0
            selector = selector * runtime_selector
        return self.sgr_parent_weight * selector.view(1, -1, 1, 1)

    def set_sgr_inference_parents(self, variable_indices=None):
        if variable_indices is None:
            self.sgr_runtime_parent_indices = None
            return
        indices = sorted({int(index) for index in variable_indices})
        for variable_index in indices:
            if variable_index < 0 or variable_index >= self.n_vars:
                raise ValueError("SGR parent index is out of range")
            if variable_index == self.uncle_target_index:
                raise ValueError("SGR target cannot be selected as a parent")
        self.sgr_runtime_parent_indices = tuple(indices)

    def configure_residual_finetune(self):
        """Freeze the trained PCFB and DARA and optimize only SGR parameters."""
        self.requires_grad_(False)
        trainable = []
        for name, parameter in self.named_parameters():
            if name.startswith("sgr_"):
                parameter.requires_grad_(True)
                trainable.append(name)
        if not trainable:
            raise RuntimeError("SGR residual finetune found no parameters")
        print("[SGR] frozen-base residual finetune")
        print("[SGR] trainable parameter tensors:", len(trainable))

    def _normalize_window(self, x):
        mean = x.mean(dim=1, keepdim=True).detach()
        std = x.std(dim=1, keepdim=True, unbiased=False).detach()
        return (x - mean) / (std + self.eps)

    def _latest_lag_features(self, x, mask):
        values = torch.stack(
            [x[:, -lag, :] for lag in self.sgr_lags],
            dim=2,
        )
        masks = torch.stack(
            [mask[:, -lag, :] for lag in self.sgr_lags],
            dim=2,
        )
        return torch.stack([values, masks], dim=-1)

    def _history_lag_features(self, x, mask):
        start = self.sgr_lags[-1]
        end = x.size(1)
        values = torch.stack(
            [x[:, start - lag:end - lag, :] for lag in self.sgr_lags],
            dim=3,
        )
        masks = torch.stack(
            [
                mask[:, start - lag:end - lag, :]
                for lag in self.sgr_lags
            ],
            dim=3,
        )
        return torch.stack([values, masks], dim=-1)

    def _encode_features(self, features):
        parent_weight = self._effective_parent_weight()
        if features.dim() == 4:
            target_features = features[:, self.uncle_target_index]
            self_pre = torch.einsum(
                "blc,hlc->bh",
                target_features,
                self.sgr_self_weight,
            )
            parent_pre = torch.einsum(
                "bvlc,hvlc->bh",
                features,
                parent_weight,
            )
        elif features.dim() == 5:
            target_features = features[:, :, self.uncle_target_index]
            self_pre = torch.einsum(
                "bnlc,hlc->bnh",
                target_features,
                self.sgr_self_weight,
            )
            parent_pre = torch.einsum(
                "bnvlc,hvlc->bnh",
                features,
                parent_weight,
            )
        else:
            raise ValueError("unexpected SGR feature rank")

        self_hidden = F.gelu(self_pre + self.sgr_self_bias)
        # No parent-side bias: a zero parent signal must produce zero output.
        parent_hidden = F.gelu(parent_pre)
        return self_hidden, parent_hidden

    def _parent_coverage(self, features):
        masks = features[..., 1]
        weight = self._effective_parent_weight().detach()
        strengths = torch.sqrt(
            weight.float().square().sum(dim=(0, 3)).clamp_min(self.eps)
        ).to(device=masks.device, dtype=masks.dtype)
        total = strengths.sum()
        if float(total.cpu()) <= self.eps:
            selector = self.sgr_parent_selector.to(
                device=masks.device,
                dtype=masks.dtype,
            )
            strengths = selector.view(-1, 1).expand_as(strengths)
            total = strengths.sum()
        return (
            masks * strengths.unsqueeze(0)
        ).sum(dim=(1, 2)) / total.clamp_min(self.eps)

    def _sgr_residual(self, x, mask):
        x, mask = self._audit_branch_inputs(x, mask)
        features = self._latest_lag_features(x, mask)
        self_hidden, parent_hidden = self._encode_features(features)

        parent_coverage = self._parent_coverage(features).clamp(0.0, 1.0)
        target_coverage = features[
            :, self.uncle_target_index, :, 1
        ].mean(dim=1).clamp(0.0, 1.0)

        condition = torch.sigmoid(self.sgr_condition(self_hidden))
        interaction = parent_hidden * condition
        raw_residual = self.sgr_residual_head(interaction)

        gate_features = torch.cat(
            [
                self_hidden,
                parent_hidden,
                parent_coverage.unsqueeze(1),
                target_coverage.unsqueeze(1),
            ],
            dim=1,
        )
        learned_gate = torch.sigmoid(self.sgr_gate_head(gate_features))
        support_gate = (
            learned_gate
            * parent_coverage.unsqueeze(1)
            * target_coverage.unsqueeze(1)
        )

        scale = self.sgr_max_scale * torch.tanh(self.sgr_scale_raw)
        bounded = self.sgr_correction_clip * torch.tanh(
            raw_residual / self.sgr_correction_clip
        )
        correction = scale * support_gate * bounded

        self._record_sgr_diagnostics(
            scale,
            support_gate,
            correction,
            parent_coverage,
            target_coverage,
        )
        return correction.unsqueeze(-1)

    def set_sgr_audit_permutation(self, variable_index=-1, block_size=7):
        variable_index = int(variable_index)
        block_size = int(block_size)
        if variable_index < -1 or variable_index >= self.n_vars:
            raise ValueError("SGR audit variable index is out of range")
        if variable_index == self.uncle_target_index:
            raise ValueError("SGR audit cannot permute the target variable")
        if block_size < 1:
            raise ValueError("SGR audit block size must be positive")
        self.sgr_audit_variable = variable_index
        self.sgr_audit_block_size = block_size

    def _audit_branch_inputs(self, x, mask):
        variable_index = int(self.sgr_audit_variable)
        if variable_index < 0:
            return x, mask
        if self.training:
            raise RuntimeError(
                "SGR block permutation is restricted to evaluation mode"
            )

        time_index = torch.arange(x.size(1), device=x.device)
        blocks = list(
            torch.split(time_index, self.sgr_audit_block_size)
        )
        permutation = torch.cat(list(reversed(blocks)), dim=0)

        audited_x = x.clone()
        audited_mask = mask.clone()
        audited_x[:, :, variable_index] = x.index_select(
            1,
            permutation,
        )[:, :, variable_index]
        audited_mask[:, :, variable_index] = mask.index_select(
            1,
            permutation,
        )[:, :, variable_index]
        return audited_x, audited_mask

    def _group_penalty(self):
        weight = self._effective_parent_weight()
        variable_groups = torch.sqrt(
            weight.square().sum(dim=(0, 2, 3)).clamp_min(self.eps)
        )
        lag_groups = torch.sqrt(
            weight.square().sum(dim=(0, 1, 3)).clamp_min(self.eps)
        )
        return (
            self.sgr_var_lambda * variable_groups.sum()
            + self.sgr_lag_lambda * lag_groups.sum()
        )

    def compute_training_auxiliary_loss(self, x, input_mask=None):
        if self.sgr_disable:
            return x.sum() * 0.0

        mask = self._prepare_input_mask(x, input_mask)
        x_norm = self._normalize_window(x)
        features = self._history_lag_features(x_norm, mask)
        self_hidden, parent_hidden = self._encode_features(features)

        self_prediction = self.sgr_self_history_head(
            self_hidden
        ).squeeze(-1)
        parent_increment = self.sgr_parent_history_head(
            parent_hidden
        ).squeeze(-1)
        full_prediction = self_prediction + parent_increment

        start = self.sgr_lags[-1]
        true_target = x_norm[:, start:, self.uncle_target_index]
        valid = mask[:, start:, self.uncle_target_index]
        observed = valid.sum()
        if float(observed.detach().cpu()) <= 0.0:
            return self._group_penalty()

        full_point_loss = F.smooth_l1_loss(
            full_prediction,
            true_target,
            reduction="none",
        )
        self_point_loss = F.smooth_l1_loss(
            self_prediction,
            true_target,
            reduction="none",
        )
        full_loss = (full_point_loss * valid).sum() / observed
        self_loss = (self_point_loss * valid).sum() / observed
        return (
            full_loss
            + self.sgr_self_loss_weight * self_loss
            + self._group_penalty()
        )

    @torch.no_grad()
    def _record_sgr_diagnostics(
        self,
        scale,
        gate,
        correction,
        parent_coverage,
        target_coverage,
    ):
        batch_size = gate.size(0)
        values = torch.stack(
            [
                scale.float(),
                gate.float().mean(),
                torch.sqrt(
                    correction.float().square().mean().clamp_min(0.0)
                ),
                parent_coverage.float().mean(),
                target_coverage.float().mean(),
            ]
        ).to(self._sgr_diag_sum.device, dtype=torch.float64)
        self._sgr_diag_sum.add_(values * batch_size)
        self._sgr_diag_count.add_(batch_size)

    @torch.no_grad()
    def reset_dara_diagnostics(self):
        super().reset_dara_diagnostics()
        self._sgr_diag_sum.zero_()
        self._sgr_diag_count.zero_()

    @torch.no_grad()
    def sgr_support_diagnostics(self):
        weight = self._effective_parent_weight().float()
        variable_strength = torch.sqrt(
            weight.square().sum(dim=(0, 2, 3)).clamp_min(0.0)
        )
        lag_strength = torch.sqrt(
            weight.square().sum(dim=(0, 1, 3)).clamp_min(0.0)
        )
        variable_total = variable_strength.sum().clamp_min(self.eps)
        lag_total = lag_strength.sum().clamp_min(self.eps)
        return {
            "target_index": int(self.uncle_target_index),
            "lags": list(self.sgr_lags),
            "variable_strength_normalized": (
                variable_strength / variable_total
            ).cpu().tolist(),
            "lag_strength_normalized": (
                lag_strength / lag_total
            ).cpu().tolist(),
        }

    @torch.no_grad()
    def dara_diagnostics(self):
        diagnostics = super().dara_diagnostics()
        count = float(self._sgr_diag_count.item())
        if count <= 0.0:
            diagnostics["sgr_disabled"] = True
            return diagnostics

        values = self._sgr_diag_sum / count
        diagnostics.update(
            {
                "sgr_disabled": False,
                "sgr_scale": float(values[0].cpu()),
                "sgr_gate_mean": float(values[1].cpu()),
                "sgr_correction_rms": float(values[2].cpu()),
                "sgr_parent_coverage_mean": float(values[3].cpu()),
                "sgr_target_coverage_mean": float(values[4].cpu()),
                "sgr_support": self.sgr_support_diagnostics(),
            }
        )
        return diagnostics

    def forward(
        self,
        x,
        input_mask=None,
        return_rep=False,
    ):
        if self.sgr_disable:
            return super().forward(
                x,
                input_mask=input_mask,
                return_rep=return_rep,
            )

        mask = self._prepare_input_mask(x, input_mask)

        if self.rev_in:
            x = self.revin_layer(x, "norm")

        expert_x = x
        if self.base_causal_fusion:
            effective_var_mat = self._effective_var_mat().to(
                device=x.device,
                dtype=x.dtype,
            )
            alpha = self._alpha().to(device=x.device, dtype=x.dtype)
            causal_message = self._causal_message(x, effective_var_mat)
            if alpha.ndim == 0:
                x = x + alpha * causal_message
            else:
                x = x + alpha.view(1, 1, -1) * causal_message

        result = self.decoder(x, return_rep=return_rep)
        if return_rep:
            base_outputs, reps = result
        else:
            base_outputs = result

        sgr_correction = self._sgr_residual(expert_x, mask)
        if self.dara_disable:
            correction = sgr_correction
        else:
            correction = (
                self._branch_residuals(expert_x, mask)
                + sgr_correction
            )

        selector = self.target_selector.to(
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, -1)
        outputs = [
            output
            + correction[:, :output.size(1), :] * selector
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
