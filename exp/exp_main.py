from models.sira import Model
from logging import raiseExceptions
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, ema_update, mean_filter
from utils.metrics import metric
from thop import profile

import torch
import torch.nn as nn
from torch import optim
# from torch.optim import lr_scheduler 
import pytorch_warmup

import os
from utils.experiment_options import experiment_option
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import copy

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.vali_target_only = self._env_flag("VALI_TARGET_ONLY")
        self.test_target_only = self._env_flag("TEST_TARGET_ONLY")
        self.train_target_only = self._env_flag("TRAIN_TARGET_ONLY")
        # This switch is deliberately restricted to the *model input*.  It
        # supports a preprocessing ablation without changing which original
        # target observations contribute to training loss or evaluation.
        self.suppress_input_observation_mask = self._env_flag(
            "SUPPRESS_INPUT_OBSERVATION_MASK"
        )
        print("VALI_TARGET_ONLY:", self.vali_target_only)
        print("TEST_TARGET_ONLY:", self.test_target_only)
        print("TRAIN_TARGET_ONLY:", self.train_target_only)
        print(
            "SUPPRESS_INPUT_OBSERVATION_MASK:",
            self.suppress_input_observation_mask,
        )

        _, self.train_loader = self._get_data(flag='train')
        self.vali_data, self.vali_loader = self._get_data(flag='val')
        if getattr(self.args, "validation_only", False):
            self.test_data = None
            self.test_loader = None
            print(
                "[Exp_Main] validation-only mode: "
                "test loader is not constructed"
            )
        else:
            self.test_data, self.test_loader = self._get_data(
                flag='test'
            )
        B, T, V = next(iter(self.train_loader))[0].shape
        L = 1
        self.d_inputs = [T, L, V]
        self.model = self._build_model().to(self.device)
        self._initialize_model_for_finetune()

        input = torch.zeros(1, T, V).to(self.device)
        flops, params = profile(self.model, inputs=(input, ))
        print('Flops: % .4fG'%(flops / 1e9))
        print('params: % .4fM'% (params / 1e6)) 

    def _build_model(self):
        model = Model(self.args, d_inputs=self.d_inputs).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _initialize_model_for_finetune(self):
        checkpoint_path = os.environ.get(
            "INIT_CHECKPOINT",
            "",
        ).strip()
        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        if checkpoint_path:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"INIT_CHECKPOINT does not exist: {checkpoint_path}"
                )
            state_dict = torch.load(
                checkpoint_path,
                map_location=self.device,
            )
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            load_result = model_core.load_state_dict(
                state_dict,
                strict=False,
            )
            allowed_missing = {
                "horizon_scale_raw",
                "_chsrf_v2_diag_sum",
                "_chsrf_v2_diag_count",
            }
            allowed_missing_prefixes = tuple(
                getattr(
                    model_core,
                    "finetune_allowed_missing_prefixes",
                    lambda: (),
                )()
            )
            missing = set(load_result.missing_keys)
            unexpected = {
                key
                for key in load_result.unexpected_keys
                if (
                    key not in {"total_ops", "total_params"}
                    and not key.endswith(
                        (".total_ops", ".total_params")
                    )
                )
            }
            disallowed_missing = {
                key
                for key in missing - allowed_missing
                if not key.startswith(allowed_missing_prefixes)
            }
            if disallowed_missing or unexpected:
                raise RuntimeError(
                    "Incompatible initialization checkpoint. "
                    f"Missing: {sorted(disallowed_missing)}; "
                    f"unexpected: {sorted(unexpected)}"
                )
            print(
                "[Exp_Main] initialized model from:",
                checkpoint_path,
            )

        if self._env_flag("SIRA_FREEZE_BASE"):
            configure = getattr(
                model_core,
                "configure_residual_finetune",
                None,
            )
            if configure is None:
                raise RuntimeError(
                    "SIRA_FREEZE_BASE requires a compatible model"
                )
            configure()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag, pred_len=None)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _env_flag(self, name):
        return experiment_option(name, "0").lower() in ("1", "true", "yes", "on")

    def _metric_start_dim(self, target_only=False):
        if target_only or self.args.features == 'MS':
            return -1
        return 0


    def _unpack_batch(self, batch):
        if len(batch) == 6:
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_x_mask,
                batch_y_mask,
            ) = batch

        elif len(batch) == 5:
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
                batch_y_mask,
            ) = batch

            # Backward compatibility for old masked datasets.
            batch_x_mask = torch.ones_like(batch_x)

        elif len(batch) == 4:
            (
                batch_x,
                batch_y,
                batch_x_mark,
                batch_y_mark,
            ) = batch

            batch_x_mask = torch.ones_like(batch_x)
            batch_y_mask = torch.ones_like(batch_y)

        else:
            raise ValueError(
                "Expected a four-item, five-item, or "
                f"six-item batch, but received {len(batch)} items"
            )

        return (
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
            batch_x_mask,
            batch_y_mask,
        )

    def _forward_sira(
        self,
        model,
        batch_x,
        batch_x_mask=None,
        return_rep=False,
    ):
        model_core = (
            model.module
            if hasattr(model, "module")
            else model
        )

        kwargs = {
            "return_rep": return_rep,
        }

        if getattr(
            model_core,
            "uses_input_observation_mask",
            False,
        ):
            kwargs["input_mask"] = self._model_input_mask(
                batch_x_mask
            )

        return model(
            batch_x,
            **kwargs,
        )

    def _model_input_mask(self, batch_x_mask):
        """Return the observation semantics exposed to the model only.

        `batch_y_mask` remains untouched everywhere: it still defines the
        observed-label loss, validation metric, and test metric.
        """
        if (
            self.suppress_input_observation_mask
            and batch_x_mask is not None
        ):
            return torch.ones_like(batch_x_mask)
        return batch_x_mask

    @staticmethod
    def _masked_mse(pred, true, mask):
        mask = mask.to(
            device=pred.device,
            dtype=pred.dtype,
        )

        observed = mask.sum()
        if observed.item() <= 0:
            return None

        squared_error = (pred - true).pow(2)
        return (
            squared_error * mask
        ).sum() / observed

    def _update_model_ema(self, model_pre, model, decay=0.998):
        return ema_update(model_pre, model, decay)

    def vali(self, vali_data, vali_loader, criterion):
        total_squared_error = 0.0
        total_observed = 0.0
        export_dir = os.environ.get(
            "VALIDATION_EXPORT_DIR",
            "",
        ).strip()
        export_preds = []
        export_trues = []
        export_masks = []
        # evaluate with the smoothed (EMA) model only
        self.model_pre.eval()
        self.model.eval()
        iter_count = 0
        time_now = time.time()

        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                iter_count += 1

                (
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    batch_x_mask,
                    batch_y_mask,
                ) = self._unpack_batch(batch)

                batch_x = batch_x.float().to(self.device)
                batch_x_mask = (
                    batch_x_mask.float().to(self.device)
                )
                batch_y = batch_y.float()
                batch_y_mask = batch_y_mask.float()

                batch_x_mark = (
                    batch_x_mark.float().to(self.device)
                )
                batch_y_mark = (
                    batch_y_mark.float().to(self.device)
                )

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    raiseExceptions ('No amp Implements')
                else:
                    outputs = self._forward_sira(
                        self.model_pre,
                        batch_x,
                        batch_x_mask,
                    )
                
                if (i + 1) % 500 == 0:
                    speed = (time.time() - time_now) / iter_count
                    print('\tinference speed: {:.4f}s/iter'.format(speed))
                    iter_count = 0
                    time_now = time.time()

                force_target_vali = self._env_flag(
                    "FORCE_TARGET_VALI"
                )
                f_dim = (
                    -1
                    if force_target_vali
                    else self._metric_start_dim(
                        self.vali_target_only
                    )
                )

                pred = (
                    outputs[-1][
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]
                    .detach()
                    .cpu()
                )

                true = batch_y[
                    :,
                    -self.args.pred_len:,
                    f_dim:,
                ].detach().cpu()

                mask = batch_y_mask[
                    :,
                    -self.args.pred_len:,
                    f_dim:,
                ].detach().cpu()

                mask = mask.to(dtype=pred.dtype)
                if export_dir:
                    export_preds.append(pred.numpy())
                    export_trues.append(true.numpy())
                    export_masks.append(mask.numpy())
                observed = float(mask.sum().item())

                if observed > 0:
                    squared_error = (
                        (pred - true).pow(2) * mask
                    ).sum()

                    total_squared_error += float(
                        squared_error.item()
                    )
                    total_observed += observed
        # plot loss
    
        if total_observed <= 0:
            raise RuntimeError(
                "Validation contains no observed labels"
            )

        global_masked_mse = (
            total_squared_error / total_observed
        )

        if export_dir:
            if not export_preds:
                raise RuntimeError(
                    "Validation export contains no prediction batches"
                )
            os.makedirs(export_dir, exist_ok=True)
            np.save(
                os.path.join(export_dir, "pred.npy"),
                np.concatenate(export_preds, axis=0),
            )
            np.save(
                os.path.join(export_dir, "true.npy"),
                np.concatenate(export_trues, axis=0),
            )
            np.save(
                os.path.join(export_dir, "mask.npy"),
                np.concatenate(export_masks, axis=0),
            )
            print("validation_export_dir:", export_dir)

        # self.model.train()
        return float(global_masked_mse)
    
    def time_freq_loss(
        self,
        y_hat,
        y,
        alpha_freq,
        mask=None,
    ):
        if mask is None:
            mask = torch.ones_like(y)

        mask = mask.to(
            device=y_hat.device,
            dtype=y_hat.dtype,
        )

        residual = (y_hat - y) * mask
        observed_total = mask.sum().clamp_min(1.0)

        # ReNF previously optimized masked MAE regardless of --loss.
        # Keep the old behaviour as the default, while allowing a
        # controlled MSE-alignment experiment.
        loss_mode = getattr(
            self,
            "sira_time_loss_mode",
            experiment_option(
                "SIRA_TIME_LOSS",
                "mae",
            ).strip().lower(),
        )

        if loss_mode == "mae":
            time_loss = (
                residual.abs().sum()
                / observed_total
            )
        elif loss_mode == "mse":
            squared_error = (
                (y_hat - y).pow(2)
                * mask
            )

            time_loss = (
                squared_error.sum()
                / observed_total
            )
        else:
            raise ValueError(
                "SIRA_TIME_LOSS must be 'mae' or 'mse', "
                f"but received {loss_mode!r}"
            )

        # Missing positions contribute zero residual.
        # Per-series scaling prevents heavily masked
        # sequences from being artificially down-weighted.
        observed_per_series = mask.sum(
            dim=1,
            keepdim=True,
        )

        valid_series = (
            observed_per_series > 0
        ).to(y_hat.dtype)

        scaled_residual = residual * (
            float(y_hat.size(1))
            / observed_per_series.clamp_min(1.0)
        )

        freq_error = torch.fft.rfft(
            scaled_residual.float(),
            dim=1,
        ).abs()

        valid_freq = valid_series.expand(
            -1,
            freq_error.size(1),
            -1,
        ).to(freq_error.dtype)

        freq_loss = (
            (freq_error * valid_freq).sum()
            / valid_freq.sum().clamp_min(1.0)
        )

        return (
            alpha_freq * freq_loss
            + (1 - alpha_freq) * time_loss
        )

    # def time_freq_loss2(self, y_hat, y, alpha_freq):
    #     freq_loss = ((torch.fft.rfft(y_hat, dim=1) - torch.fft.rfft(y, dim=1)).abs()**2).mean()
    #     # weights = (20. / (torch.arange(y_hat.shape[1], dtype=torch.float32, device=y_hat.device) + 1.0)).unsqueeze(0).unsqueeze(-1)
    #     time_loss = ((y_hat - y)**2).mean()
    #     return alpha_freq * freq_loss + (1 - alpha_freq) * time_loss

    def train(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(self.train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        criterion_val = nn.MSELoss()

        self.sira_time_loss_mode = experiment_option(
            "SIRA_TIME_LOSS",
            "mae",
        ).strip().lower()

        if self.sira_time_loss_mode not in {
            "mae",
            "mse",
        }:
            raise ValueError(
                "SIRA_TIME_LOSS must be 'mae' or 'mse', "
                f"but received {self.sira_time_loss_mode!r}"
            )

        print(
            "[Exp_Main] SIRA_TIME_LOSS:",
            self.sira_time_loss_mode,
        )

        target_aux_loss_weight = float(
            os.environ.get(
                "TARGET_AUX_LOSS_WEIGHT",
                "0.0",
            )
        )
        print(
            "[Exp_Main] target auxiliary loss weight:",
            target_aux_loss_weight,
        )
        print(
            "[Exp_Main] force target validation:",
            self._env_flag("FORCE_TARGET_VALI"),
        )

        dara_history_loss_weight = float(
            experiment_option(
                "DARA_HISTORY_LOSS_WEIGHT",
                "0.0",
            )
        )
        if dara_history_loss_weight < 0.0:
            raise ValueError(
                "DARA_HISTORY_LOSS_WEIGHT must be non-negative"
            )
        print(
            "[Exp_Main] DARA history loss weight:",
            dara_history_loss_weight,
        )

        dara_residual_loss_weight = float(
            experiment_option(
                "DARA_RESIDUAL_LOSS_WEIGHT",
                "0.0",
            )
        )
        if dara_residual_loss_weight < 0.0:
            raise ValueError(
                "DARA_RESIDUAL_LOSS_WEIGHT must be non-negative"
            )
        print(
            "[Exp_Main] DARA residual loss weight:",
            dara_residual_loss_weight,
        )

        dara_reliability_loss_weight = float(
            experiment_option(
                "DARA_RELIABILITY_LOSS_WEIGHT",
                "0.0",
            )
        )
        if dara_reliability_loss_weight < 0.0:
            raise ValueError(
                "DARA_RELIABILITY_LOSS_WEIGHT must be non-negative"
            )
        print(
            "[Exp_Main] DARA reliability loss weight:",
            dara_reliability_loss_weight,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(model_optim, step_size=2, gamma=self.args.gamma)
        self.model.train()
        self.model_pre = copy.deepcopy(self.model).requires_grad_(False)
        valid_losses = []
        test_losses = []
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            for i, batch in enumerate(self.train_loader):
                iter_count += 1
                model_optim.zero_grad()

                (
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    batch_x_mask,
                    batch_y_mask,
                ) = self._unpack_batch(batch)

                batch_x = batch_x.float().to(self.device)
                batch_x_mask = (
                    batch_x_mask.float().to(self.device)
                )
                batch_y = batch_y.float().to(self.device)
                batch_y_mask = (
                    batch_y_mask.float().to(self.device)
                )

                batch_x_mark = (
                    batch_x_mark.float().to(self.device)
                )
                batch_y_mark = (
                    batch_y_mark.float().to(self.device)
                )

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    pass
                else:
                    outputs = self._forward_sira(
                        self.model,
                        batch_x,
                        batch_x_mask,
                    )
                    n = len(outputs)
                    k = self.args.pred_len // n
                    f_dim = self._metric_start_dim(self.train_target_only)

                    # HRL coefficients
                    # lamb = 0.8 # set to 0 to disable
                    # coefs_L = torch.ones_like(batch_y).cuda()
                    # n_c = 4 # better be a factor of pred_len
                    # for j in range(n_c):
                    #     m = self.args.pred_len // n_c
                    #     end = (j + 1) * m
                    #     coefs_L[:, j:end, f_dim:] *= (0.5 + lamb * (n_c - 1 - j))

                    loss = 0.

                    # Stage-loss allocation.
                    #
                    # legacy:
                    #     20, 10, 20/3
                    #
                    # custom example:
                    #     STAGE_LOSS_WEIGHTS=0.25,0.5,1.0
                    #
                    # Custom weights are rescaled to preserve the total
                    # legacy loss magnitude, so the comparison does not
                    # accidentally change the effective learning rate.
                    stage_weight_spec = os.environ.get(
                        "STAGE_LOSS_WEIGHTS",
                        "legacy",
                    ).strip().lower()

                    legacy_stage_weights = [
                        20.0 / (stage_index + 1)
                        for stage_index in range(n)
                    ]

                    if stage_weight_spec == "legacy":
                        stage_weights = legacy_stage_weights
                    else:
                        raw_stage_weights = [
                            float(value.strip())
                            for value in stage_weight_spec.split(",")
                            if value.strip()
                        ]

                        if len(raw_stage_weights) != n:
                            raise ValueError(
                                "STAGE_LOSS_WEIGHTS must contain "
                                f"{n} values, but received "
                                f"{raw_stage_weights}"
                            )

                        if any(
                            value < 0.0
                            for value in raw_stage_weights
                        ):
                            raise ValueError(
                                "STAGE_LOSS_WEIGHTS must be non-negative"
                            )

                        raw_sum = sum(raw_stage_weights)

                        if raw_sum <= 0.0:
                            raise ValueError(
                                "STAGE_LOSS_WEIGHTS must have "
                                "a positive sum"
                            )

                        legacy_sum = sum(
                            legacy_stage_weights
                        )

                        stage_weights = [
                            value
                            * legacy_sum
                            / raw_sum
                            for value in raw_stage_weights
                        ]

                    if i == 0 and epoch == 0:
                        print(
                            "[Exp_Main] stage loss specification:",
                            stage_weight_spec,
                        )
                        print(
                            "[Exp_Main] effective stage weights:",
                            stage_weights,
                        )

                    model_for_target = (
                        self.model.module
                        if hasattr(self.model, "module")
                        else self.model
                    )

                    configured_target_index = int(
                        getattr(
                            model_for_target,
                            "uncle_target_index",
                            -1,
                        )
                    )

                    for j in range(n):
                        output_j = outputs[j]
                        output_length = (j + 1) * k

                        true_j = batch_y[
                            :,
                            :output_length,
                            f_dim:,
                        ]

                        mask_j = batch_y_mask[
                            :,
                            :output_length,
                            f_dim:,
                        ]

                        stage_loss = self.time_freq_loss(
                            output_j,
                            true_j,
                            self.args.alpha_freq,
                            mask=mask_j,
                        )

                        if target_aux_loss_weight > 0.0:
                            if output_j.size(-1) == 1:
                                pred_target = output_j
                            else:
                                output_target_index = max(
                                    0,
                                    min(
                                        configured_target_index,
                                        output_j.size(-1) - 1,
                                    ),
                                )
                                pred_target = output_j[
                                    :,
                                    :,
                                    output_target_index:
                                    output_target_index + 1,
                                ]

                            true_target_index = max(
                                0,
                                min(
                                    configured_target_index,
                                    batch_y.size(-1) - 1,
                                ),
                            )

                            true_target = batch_y[
                                :,
                                :output_length,
                                true_target_index:
                                true_target_index + 1,
                            ]

                            target_mask = batch_y_mask[
                                :,
                                :output_length,
                                true_target_index:
                                true_target_index + 1,
                            ]

                            target_stage_loss = (
                                self.time_freq_loss(
                                    pred_target,
                                    true_target,
                                    self.args.alpha_freq,
                                    mask=target_mask,
                                )
                            )

                            stage_loss = (
                                stage_loss
                                + target_aux_loss_weight
                                * target_stage_loss
                            )

                        loss += (
                            stage_weights[j]
                            * stage_loss
                        )

                    if (
                        dara_history_loss_weight > 0.0
                        and hasattr(
                            model_for_target,
                            "compute_training_auxiliary_loss",
                        )
                    ):
                        dara_history_loss = (
                            model_for_target
                            .compute_training_auxiliary_loss(
                                batch_x,
                                self._model_input_mask(
                                    batch_x_mask
                                ),
                            )
                        )
                        loss = (
                            loss
                            + dara_history_loss_weight
                            * dara_history_loss
                        )

                        if i == 0 and epoch == 0:
                            print(
                                "[Exp_Main] initial DARA "
                                "history loss:",
                                float(
                                    dara_history_loss
                                    .detach()
                                    .cpu()
                                ),
                            )

                    if dara_residual_loss_weight > 0.0:
                        residual_loss_fn = getattr(
                            model_for_target,
                            "compute_residual_supervision_loss",
                            None,
                        )
                        if residual_loss_fn is None:
                            raise RuntimeError(
                                "DARA_RESIDUAL_LOSS_WEIGHT requires "
                                "a compatible model"
                            )
                        dara_residual_loss = residual_loss_fn(
                            batch_y,
                            batch_y_mask,
                        )
                        loss = (
                            loss
                            + dara_residual_loss_weight
                            * dara_residual_loss
                        )
                        if i == 0 and epoch == 0:
                            print(
                                "[Exp_Main] initial DARA "
                                "residual loss:",
                                float(
                                    dara_residual_loss
                                    .detach()
                                    .cpu()
                                ),
                            )

                    if dara_reliability_loss_weight > 0.0:
                        reliability_loss_fn = getattr(
                            model_for_target,
                            "compute_reliability_supervision_loss",
                            None,
                        )
                        if reliability_loss_fn is None:
                            raise RuntimeError(
                                "DARA_RELIABILITY_LOSS_WEIGHT "
                                "requires a compatible model"
                            )
                        dara_reliability_loss = reliability_loss_fn(
                            batch_y,
                            batch_y_mask,
                        )
                        loss = (
                            loss
                            + dara_reliability_loss_weight
                            * dara_reliability_loss
                        )
                        if i == 0 and epoch == 0:
                            print(
                                "[Exp_Main] initial DARA "
                                "reliability loss:",
                                float(
                                    dara_reliability_loss
                                    .detach()
                                    .cpu()
                                ),
                            )
                    # loss = self.time_freq_loss(outputs[-1], batch_y[:, :, f_dim:], self.args.alpha_freq)
                    train_loss.append(loss.item())
  
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    pass
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    model_optim.step()
                    with torch.no_grad():
                        self.model_pre = self._update_model_ema(self.model_pre, self.model, decay=self.args.r_ema)
                # store the weight of each layer
                # if (epoch == self.args.train_epochs - 1) and (i == train_steps - 5):
                #     weights = []
                #     import matplotlib.pyplot as plt
                #     for name, param in self.model.named_parameters():
                #         if 'weight' in name:
                #             weights.append(torch.norm(param.data.cpu(), p=2))
                #     plt.plot(weights)
                #     plt.savefig('weights.png')
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.mean(train_loss)
            vali_loss = self.vali(self.vali_data, self.vali_loader, criterion_val)  # use the same indicator as in training phase
            valid_losses.append(vali_loss)
            if getattr(self.args, "validation_only", False):
                print(
                    "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} "
                    "Vali Loss: {3:.7f}".format(
                        epoch + 1,
                        train_steps,
                        train_loss,
                        vali_loss,
                    )
                )
            else:
                test_loss = self.vali(
                    self.test_data,
                    self.test_loader,
                    criterion_val,
                )
                test_losses.append(test_loss)
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model_pre, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            scheduler.step()
            print('Updating learning rate to {:.6f}'.format(scheduler.get_last_lr()[0]))
        # import matplotlib.pyplot as plt
        # np.save('./toy_exp/valid_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.npy', valid_losses)
        # np.save('./toy_exp/test_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.npy', test_losses)
        # plt.plot()
        # plt.savefig('./toy_exp/valid_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.pdf')
        # plt.clf()
        # plt.plot()
        # plt.savefig('./toy_exp/test_loss_' + self.args.model_id + '_' + str(self.args.r_ema) + '.pdf')
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.best_validation_loss = float(
            early_stopping.val_loss_min
        )
        print(
            "best_validation_mse:"
            f"{self.best_validation_loss}"
        )

        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        support_diagnostics = getattr(
            model_core,
            "sgr_support_diagnostics",
            None,
        )
        if callable(support_diagnostics):
            print(
                "[SGR-v1] learned support:",
                support_diagnostics(),
            )

        return self.model

    def post_process(self, outputs, batch_y, iter, strategy='average', plot=True):
        # Keep tensors on GPU for computation, store original device
        device = batch_y.device

        if strategy != 'average':
            raise ValueError(
                "Oracle truth-dependent post-processing "
                "has been disabled. Use strategy='average'."
            )
        
        n = len(outputs)
        k = self.args.pred_len // n
    
        if strategy == 'average':
            pred_com = torch.zeros_like(batch_y)
            weight_sum = torch.zeros_like(batch_y)
            
            for i, output in enumerate(outputs):
                # Each output has length (i+1) * k where k = pred_len // n
                output_len = output.shape[1]
                target_len = min(output_len, batch_y.shape[1])
                
                # Add this output's contribution to the overlapping region
                pred_com[:, :target_len, :] += output[:, :target_len, :]
                weight_sum[:, :target_len, :] += 1.0
            
            # Normalize by the number of outputs that contributed to each position
            pred_com = torch.where(weight_sum > 0, pred_com / weight_sum, pred_com)
        else:
            # Optimal combination
            if n == 0:
                pred_com = torch.zeros_like(batch_y)
            else:
                pred_com = torch.zeros_like(batch_y)
                seq_len = min(outputs[0].shape[1], batch_y.shape[1])
                pred_com[:, :seq_len, :] = outputs[0][:, :seq_len, :]
                
                # Compare with subsequent outputs one by one
                for i in range(1, n):
                    output = outputs[i]
                    seq_len = min(output.shape[1], batch_y.shape[1])
                    
                    # Calculate errors only for the relevant sequence length
                    pred_error = torch.abs(batch_y[:, :seq_len, :] - pred_com[:, :seq_len, :])
                    output_error = torch.abs(batch_y[:, :seq_len, :] - output[:, :seq_len, :])
                    
                    # Update where output is better
                    mask = output_error < pred_error
                    pred_com[:, :seq_len, :] = torch.where(mask, output[:, :seq_len, :], pred_com[:, :seq_len, :])
        
        # Only create plot when actually needed - move to CPU only for plotting
        if plot and iter == 10:
            # Move to CPU only for plotting
            batch_y_cpu = batch_y.detach().cpu()
            pred_com_cpu = pred_com.detach().cpu()
            outputs_cpu = [output.detach().cpu() for output in outputs]
            
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot forecasts
            for i, output_cpu in enumerate(outputs_cpu):
                ax.plot(output_cpu[0, :, 0].numpy(), label=f'forecast_{i}')
            
            # Plot ground truth and combined prediction
            ax.plot(batch_y_cpu[0, :, 0].numpy(), label='true observation')
            ax.plot(pred_com_cpu[0, :, 0].numpy(), label=strategy)
            
            # Optimized styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True,
                     framealpha=0.9, loc='upper left')
            
            # Set spine properties in batch
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
            
            ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
            plt.tight_layout()
            plt.savefig(f'./forecast_visual/result_{self.args.model_id}_{iter}_{strategy}.pdf')
            plt.close()
            
        return pred_com

    def evaluate_validation_checkpoint(self, setting):
        """Load one checkpoint and evaluate validation without test access."""
        checkpoint_path = os.path.join(
            self.args.checkpoints,
            setting,
            "checkpoint.pth",
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model_pre = self.model

        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        if hasattr(model_core, "reset_dara_diagnostics"):
            model_core.reset_dara_diagnostics()

        validation_mse = self.vali(
            self.vali_data,
            self.vali_loader,
            self._select_criterion(),
        )
        print(f"checkpoint_validation_mse:{validation_mse}")

        diagnostics = getattr(
            model_core,
            "dara_diagnostics",
            None,
        )
        if callable(diagnostics):
            print(
                "[DARA] validation diagnostics:",
                diagnostics(),
            )
        return validation_mse

    def test(self, setting, test=0):
        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(
                    os.path.join(
                        "./checkpoints/" + setting,
                        "checkpoint.pth",
                    ),
                    map_location=self.device,
                )
            )

        all_preds = []
        all_trues = []
        all_masks = []

        last_reps = None
        self.model.eval()

        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        if hasattr(model_core, "reset_cmsf_diagnostics"):
            model_core.reset_cmsf_diagnostics()
        if hasattr(model_core, "reset_ccc_diagnostics"):
            model_core.reset_ccc_diagnostics()
        if hasattr(model_core, "reset_dara_diagnostics"):
            model_core.reset_dara_diagnostics()

        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                (
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    batch_x_mask,
                    batch_y_mask,
                ) = self._unpack_batch(batch)

                batch_x = batch_x.float().to(self.device)
                batch_x_mask = (
                    batch_x_mask.float().to(self.device)
                )
                batch_y = batch_y.float().to(self.device)
                batch_y_mask = batch_y_mask.float()

                if self.args.use_amp:
                    raise Exception("amp not supported")

                outputs, reps = self._forward_sira(
                    self.model,
                    batch_x,
                    batch_x_mask,
                    return_rep=True,
                )
                last_reps = reps

                f_dim = self._metric_start_dim(
                    self.test_target_only
                )

                pred = (
                    outputs[-1][
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                true = (
                    batch_y[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                mask = (
                    batch_y_mask[
                        :,
                        -self.args.pred_len:,
                        f_dim:,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                if not (
                    pred.shape
                    == true.shape
                    == mask.shape
                ):
                    raise ValueError(
                        "Prediction/label/mask shapes "
                        "do not match: "
                        f"{pred.shape}, "
                        f"{true.shape}, "
                        f"{mask.shape}"
                    )

                all_preds.append(pred)
                all_trues.append(true)
                all_masks.append(mask)

        if self.args.test_flop:
            test_params_flop(
                (
                    batch_x.shape[1],
                    batch_x.shape[2],
                )
            )
            return

        if not all_preds:
            raise RuntimeError("Test loader is empty")

        preds = np.concatenate(
            all_preds,
            axis=0,
        )
        trues = np.concatenate(
            all_trues,
            axis=0,
        )
        masks = np.concatenate(
            all_masks,
            axis=0,
        )

        observed = masks > 0.5
        observed_count = int(observed.sum())
        total_count = int(observed.size)

        if observed_count == 0:
            raise RuntimeError(
                "Test set contains no observed labels"
            )

        observed_preds = preds[observed]
        observed_trues = trues[observed]

        print(
            "Full result tensor shape:",
            trues.shape,
        )
        print(
            "Observed results shape:",
            observed_trues.shape,
        )
        print(
            "Observed labels:",
            f"{observed_count}/{total_count}",
            f"({observed_count / total_count:.6f})",
        )

        mae, mse = metric(
            observed_preds,
            observed_trues,
        )

        print(
            "mse:{}, mae:{}".format(
                mse,
                mae,
            )
        )

        if hasattr(model_core, "cmsf_diagnostics"):
            diagnostics = model_core.cmsf_diagnostics()
            if diagnostics:
                print("[CMSF-v1] test diagnostics:", diagnostics)
        if hasattr(model_core, "ccc_diagnostics"):
            diagnostics = model_core.ccc_diagnostics()
            if diagnostics:
                print("[CCC-v1] test diagnostics:", diagnostics)
        if hasattr(model_core, "dara_diagnostics"):
            diagnostics = model_core.dara_diagnostics()
            if diagnostics:
                print("[DARA] test diagnostics:", diagnostics)

        folder_path = (
            "./results/" + setting + "/"
        )
        os.makedirs(
            folder_path,
            exist_ok=True,
        )

        np.save(
            folder_path + "pred.npy",
            preds,
        )
        np.save(
            folder_path + "true.npy",
            trues,
        )
        np.save(
            folder_path + "mask.npy",
            masks,
        )
        np.save(
            folder_path + "pred_observed.npy",
            observed_preds,
        )
        np.save(
            folder_path + "true_observed.npy",
            observed_trues,
        )

        if last_reps is not None:
            rep_store = {}
            for n in range(len(last_reps)):
                rep_store[
                    "rep" + str(n)
                ] = (
                    last_reps[n]
                    .detach()
                    .cpu()
                    .numpy()
                )
            np.save(
                folder_path + "rep_store.npy",
                rep_store,
            )

        with open("result.txt", "a") as f:
            f.write(setting + "  \n")
            f.write(
                "mse:{}, mae:{}, "
                "observed:{}/{}".format(
                    mse,
                    mae,
                    observed_count,
                    total_count,
                )
            )
            f.write("\n\n")

        return
