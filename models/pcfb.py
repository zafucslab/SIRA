"""Progressive Cascaded Forecasting Backbone (PCFB)."""

import os
from utils.experiment_options import experiment_option
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from layers.RevIN import RevIN
from layers.pcfb_blocks import Decoder, ForeBlock, MidBlock


class Model(nn.Module):
    def __init__(self, configs, d_inputs, pe='zeros', learn_pe=True):
        super(Model, self).__init__()

        self.rev_in = configs.revin
        self.pred_len = configs.pred_len
        self.L = configs.level_dim
        self.n_vars = d_inputs[-1]

        self.uncle_strength = float(experiment_option("PCFB_STRENGTH", os.environ.get("PRIOR_STRENGTH", "0.02")))
        self.uncle_init_std = float(experiment_option("PCFB_INIT_STD", "0.01"))
        self.uncle_gate_power = float(experiment_option("PCFB_GATE_POWER", "1.0"))
        self.uncle_alpha_mode = experiment_option("PCFB_ALPHA_MODE", "variable").lower()
        self.uncle_alpha_init = float(experiment_option("PCFB_ALPHA_INIT", "0.001"))
        self.uncle_lag = int(experiment_option("PCFB_LAG", "1"))
        self.uncle_lag_mode = experiment_option("PCFB_LAG_MODE", "fixed").lower()
        self.uncle_var_relu = experiment_option("PCFB_VAR_RELU", "0").lower() in ("1", "true", "yes", "on")
        self.uncle_fusion_mode = experiment_option("PCFB_FUSION_MODE", "input").lower()

        # If enabled, keep only edges pointing to the target variable.
        # In the current data pipeline, _tensor_columns() moves the target column to the last variable.
        # This is safer for target-only forecasting because graph fusion only injects parent-variable
        # information into the target branch and does not perturb all other variables.
        self.uncle_target_only_fuse = experiment_option(
            "PCFB_TARGET_ONLY_FUSE", "0"
        ).lower() in ("1", "true", "yes", "on")
        self.uncle_target_index = self.n_vars - 1

        if self.uncle_lag < 1:
            raise ValueError("PCFB_LAG must be >= 1")
        if self.uncle_lag_mode not in ("fixed", "softmax"):
            raise ValueError("PCFB_LAG_MODE must be one of: fixed, softmax")
        if self.uncle_fusion_mode not in ("input", "decoder", "both"):
            raise ValueError("PCFB_FUSION_MODE must be one of: input, decoder, both")

        if self.rev_in:
            self.revin_layer = RevIN(self.n_vars, affine=True, subtract_last=False)

        decoder_cls = CausalDecoder if self.uncle_fusion_mode in ("decoder", "both") else Decoder
        self.decoder = decoder_cls(
            d_inputs,
            self.pred_len,
            n_c=configs.d_layers,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            pe=configs.pe
        )

        # Save RNG state right after Alpha-equivalent decoder initialization.
        # Extra graph-fusion parameter initialization consumes torch RNG. Restoring it
        # makes zero-control training follow the same random trajectory as Alpha.
        _uncle_rng_state_cpu = torch.get_rng_state()
        _uncle_rng_state_cuda = None
        if torch.cuda.is_available():
            try:
                _uncle_rng_state_cuda = torch.cuda.get_rng_state_all()
            except Exception:
                _uncle_rng_state_cuda = None

        gate = self._load_gate(configs)
        self.register_buffer("prior_gate", gate)

        self.var_mat = nn.Parameter(torch.empty(self.n_vars, self.n_vars, self.uncle_lag))
        nn.init.normal_(self.var_mat, mean=0.0, std=self.uncle_init_std)
        with torch.no_grad():
            diag = torch.arange(self.n_vars)
            self.var_mat[diag, diag, :].zero_()

        off_diag = torch.ones(self.n_vars, self.n_vars) - torch.eye(self.n_vars)
        self.register_buffer("off_diag", off_diag)

        alpha_ratio = min(max(self.uncle_alpha_init / max(self.uncle_strength, 1e-8), 1e-6), 1.0 - 1e-6)
        alpha_logit = math.log(alpha_ratio / (1.0 - alpha_ratio))
        if self.uncle_alpha_mode == "global":
            self.alpha_logit = nn.Parameter(torch.tensor(alpha_logit, dtype=torch.float32))
        elif self.uncle_alpha_mode == "variable":
            self.alpha_logit = nn.Parameter(torch.full((self.n_vars,), alpha_logit, dtype=torch.float32))
        else:
            raise ValueError("PCFB_ALPHA_MODE must be one of: global, variable")

        if self.uncle_lag_mode == "softmax" and self.uncle_lag > 1:
            lag_bias = torch.linspace(float(self.uncle_lag - 1), 0.0, self.uncle_lag)
            self.lag_logit = nn.Parameter(lag_bias.repeat(self.n_vars, 1))
        else:
            self.register_parameter("lag_logit", None)

        # Restore RNG state after graph-fusion parameters are initialized.
        # This keeps subsequent training randomness aligned with the base forecaster.
        torch.set_rng_state(_uncle_rng_state_cpu)
        if _uncle_rng_state_cuda is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(_uncle_rng_state_cuda)
            except Exception:
                pass
        print("[PCFB] restored RNG state after graph-fusion initialization")

        print("[PCFB] uncle_strength:", self.uncle_strength)
        print("[PCFB] uncle_init_std:", self.uncle_init_std)
        print("[PCFB] uncle_gate_power:", self.uncle_gate_power)
        print("[PCFB] uncle_alpha_mode:", self.uncle_alpha_mode)
        print("[PCFB] uncle_alpha_init:", self.uncle_alpha_init)
        print("[PCFB] uncle_lag:", self.uncle_lag)
        print("[PCFB] uncle_lag_mode:", self.uncle_lag_mode)
        print("[PCFB] uncle_var_relu:", self.uncle_var_relu)
        print("[PCFB] uncle_fusion_mode:", self.uncle_fusion_mode)
        print("[PCFB] uncle_target_only_fuse:", self.uncle_target_only_fuse)
        print("[PCFB] uncle_target_index:", self.uncle_target_index)

    def _fallback_columns(self, configs):
        target = getattr(configs, "target", None)
        features = getattr(configs, "features", "M")
        if features == "S" and target:
            return [target]

        columns = [
            "air_temperature",
            "rainfall",
            "wind_speed",
            "soil_temp_5cm",
            "soil_temp_20cm",
            "soil_moisture_5cm",
            "soil_moisture_20cm",
            "solar_radiation",
            "relative_humidity",
            "sunshine_duration",
        ]
        if target in columns:
            columns = [col for col in columns if col != target] + [target]
        return columns

    def _tensor_columns(self, configs):
        root_path = getattr(configs, "root_path", None)
        data_path = getattr(configs, "data_path", None)
        target = getattr(configs, "target", None)
        features = getattr(configs, "features", "M")
        if not (root_path and data_path and target):
            return self._fallback_columns(configs)

        csv_path = os.path.join(root_path, data_path)
        if not os.path.exists(csv_path):
            return self._fallback_columns(configs)

        raw_cols = list(pd.read_csv(csv_path, nrows=0).columns)
        if features == "S":
            return [target]
        if "date" not in raw_cols or target not in raw_cols:
            return self._fallback_columns(configs)

        cols = raw_cols.copy()
        cols.remove(target)
        cols.remove("date")
        return cols + [target]

    def _load_gate(self, configs):
        default_prior_path = "graphs/agriculture_prior.csv"
        graph_path = experiment_option("PCFB_GRAPH_PATH") or os.environ.get("PRIOR_GRAPH_PATH") or default_prior_path
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Cannot find prior/UNCLE graph: {graph_path}")

        tensor_cols = self._tensor_columns(configs)
        if len(tensor_cols) != self.n_vars:
            raise ValueError(f"UNCLE graph column count {len(tensor_cols)} does not match model variables {self.n_vars}")

        graph = pd.read_csv(graph_path, index_col=0).loc[tensor_cols, tensor_cols]
        gate = torch.tensor(graph.values, dtype=torch.float32).clamp_min(0.0)

        # Important: remove self-loops before normalization.
        # Otherwise self-loop weights enter the denominator, but are later removed
        # by off_diag in _effective_var_mat(), which weakens useful cross-variable edges.
        eye = torch.eye(gate.shape[0], dtype=gate.dtype)
        gate = gate * (1.0 - eye)

        normalize = experiment_option("PCFB_GATE_NORMALIZE", os.environ.get("PRIOR_NORMALIZE", "none")).lower()
        if normalize == "column":
            gate = gate / gate.sum(dim=0, keepdim=True).clamp_min(1e-6)
        elif normalize == "row":
            gate = gate / gate.sum(dim=1, keepdim=True).clamp_min(1e-6)
        elif normalize in ("none", "false", "0"):
            pass
        else:
            raise ValueError("PCFB_GATE_NORMALIZE must be one of: column, row, none")

        if self.uncle_gate_power != 1.0:
            gate = torch.pow(gate, self.uncle_gate_power)

        # Keep a unit diagonal in the stored gate for numerical safety / inspection.
        # The actual causal message still uses off_diag, so self-loops are not fused.
        gate = gate * (1.0 - eye) + eye

        print("[PCFB] loaded prior/UNCLE gate:", graph_path)
        print("[PCFB] tensor columns:", tensor_cols)
        print("[PCFB] gate_normalize:", normalize)
        return gate

    def _maybe_keep_target_column_only(self, effective_var_mat):
        """Optionally keep only causal edges pointing to the target variable.

        Matrix convention used by torch.matmul(x, A):
            output[..., j] = sum_i x[..., i] * A[i, j]

        Therefore column j contains parents pointing to variable j.
        Since _tensor_columns() moves the target to the last column, the target
        variable index is self.n_vars - 1.
        """
        if not self.uncle_target_only_fuse:
            return effective_var_mat

        mask = torch.zeros_like(effective_var_mat)
        mask[:, self.uncle_target_index, :] = 1.0
        return effective_var_mat * mask

    def _effective_var_mat(self):
        gate = self.prior_gate.to(dtype=self.var_mat.dtype)
        gate = gate.unsqueeze(-1)
        off_diag = self.off_diag.to(dtype=self.var_mat.dtype).unsqueeze(-1)
        effective_var_mat = self.var_mat * gate * off_diag

        if self.lag_logit is not None:
            lag_weight = torch.softmax(self.lag_logit, dim=-1).to(dtype=self.var_mat.dtype)
            effective_var_mat = effective_var_mat * lag_weight.unsqueeze(0)

        effective_var_mat = self._maybe_keep_target_column_only(effective_var_mat)
        return effective_var_mat

    def _alpha(self):
        return self.uncle_strength * torch.sigmoid(self.alpha_logit)

    def _causal_message(self, x, effective_var_mat):
        if self.uncle_lag == 1:
            return torch.matmul(x, effective_var_mat.squeeze(-1))

        lagged_x = F.pad(x.transpose(1, 2), (self.uncle_lag - 1, 0))
        lagged_x = lagged_x.unfold(2, self.uncle_lag, 1)
        causal_message = torch.einsum('bctl,cel->bte', lagged_x, effective_var_mat)

        if self.uncle_var_relu:
            causal_message = torch.relu(causal_message)
        return causal_message

    def forward(self, x, return_rep=False):
        if self.rev_in:
            x = self.revin_layer(x, 'norm')

        effective_var_mat = self._effective_var_mat()
        alpha = self._alpha().to(device=x.device, dtype=x.dtype)

        if self.uncle_fusion_mode in ("input", "both"):
            causal_message = self._causal_message(x, effective_var_mat)
            if alpha.ndim == 0:
                x = x + alpha * causal_message
            else:
                x = x + alpha.view(1, 1, -1) * causal_message

        if self.uncle_fusion_mode in ("decoder", "both"):
            result = self.decoder(
                x,
                return_rep=return_rep,
                effective_var_mat=effective_var_mat,
                alpha=alpha,
            )
        else:
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


class CausalDecoder(nn.Module):
    def __init__(self, d_inputs, d_pred, n_c=3, d_ff=256, dropout=0., pe=False, exv_use=0):
        super(CausalDecoder, self).__init__()
        self.receiver = CausalDecoderBlock(d_ff, d_inputs[0], d_pred, n_c, drop_ff=dropout)
        self.d_inputs = d_inputs
        self.pred_len = d_pred
        self.pe = pe

        W_pos = torch.empty((d_inputs[0], d_inputs[-1]))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.embedding = nn.Parameter(W_pos)

        self.exv_use = exv_use
        if exv_use:
            self.fea_proj = nn.Sequential(
                nn.Linear(4, 128, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, d_inputs[-1]),
            )

    def forward(self, x, x_mark=None, return_rep=False, effective_var_mat=None, alpha=None):
        if self.pe:
            x = x + self.embedding
        if x_mark is not None and self.exv_use:
            x = x + self.fea_proj(x_mark)

        result = self.receiver(
            x.transpose(1, 2),
            return_rep=return_rep,
            effective_var_mat=effective_var_mat,
            alpha=alpha,
        )

        if return_rep:
            ys, reps = result
        else:
            ys = result

        ys = [y.transpose(1, 2) for y in ys]

        if return_rep:
            return ys, reps
        return ys


class CausalDecoderBlock(nn.Module):
    def __init__(self, d_ff, d_in, d_out, n_c=3, drop_ff=None, drop_in=0.3):
        super(CausalDecoderBlock, self).__init__()
        self.pred_base = d_out // n_c
        self.cascade = nn.ModuleList([ForeBlock(d_in, self.pred_base, d_ff, drop_ff)])
        self.n_c = n_c

        for i in range(1, n_c):
            n_acc = i * (i + 1) // 2
            d_cat = d_in + self.pred_base * n_acc
            act = 'nonlinear' if i < n_c - 1 else 'linear'
            drop_in_layer = min(drop_in * (i + 1), 0.6)
            self.cascade.append(MidBlock(d_cat, self.pred_base * (i + 1), d_ff, drop_ff, drop_in_layer, act))

    def _causal_fuse(self, x, effective_var_mat, alpha):
        if effective_var_mat is None or alpha is None:
            return x

        # x is [B, V, T_or_T_plus_partial_predictions]. Mix across variables at
        # every decoder stage while preserving the cascade's temporal dimension.
        x_time_first = x.transpose(1, 2)

        if effective_var_mat.shape[-1] == 1:
            causal_message = torch.matmul(x_time_first, effective_var_mat.squeeze(-1))
        else:
            lag = effective_var_mat.shape[-1]
            lagged_x = F.pad(x_time_first.transpose(1, 2), (lag - 1, 0))
            lagged_x = lagged_x.unfold(2, lag, 1)
            causal_message = torch.einsum('bvtl,vel->bte', lagged_x, effective_var_mat)

        causal_message = causal_message.transpose(1, 2)

        if alpha.ndim == 0:
            return x + alpha * causal_message
        return x + alpha.view(1, -1, 1) * causal_message

    def forward(self, x0, return_rep=False, effective_var_mat=None, alpha=None):
        out = []
        x_k = x0
        reps = []

        for layer in self.cascade:
            x_in = self._causal_fuse(x_k, effective_var_mat, alpha)
            y_k, rep = layer(x_in)
            out.append(y_k)

            if return_rep:
                reps.append(rep)

            y_cat = torch.cat(out, dim=-1).detach()
            mask = 1.0
            x_k = torch.cat([x0, y_cat * mask], dim=-1)

        if return_rep:
            return out, reps
        return out
