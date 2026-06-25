"""JWST_DINO — a self-contained Lightning re-implementation of DINOv2 pre-training.

Algorithm is faithful to DINOv2 (student/teacher EMA + DINO CLS loss + iBOT patch
loss + KoLeo, with teacher centering) + image augmentation dedicated to jwst imaging,
the infrastructure is pyLightning:
DDP, bf16-mixed (no GradScaler), and manual optimization so the five DINO
schedules (lr / wd / teacher momentum / teacher temp / last-layer lr), the EMA,
per-submodule grad clipping and the last-layer freeze are all explicit.

everythin, the ViT, head and losses are all vendored under model/.
"""

import math
import os
from typing import List

import lightning as L
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW

from .dino_head import DINOHead
from .losses import DINOLoss, KoLeoLoss, iBOTPatchLoss
from .vision_transformer import VisionTransformer


def cosine_sched(step, total, base, final, warmup=0, freeze=0, start=0.0):
    """One closed-form schedule that reproduces every DINOv2 cosine curve.
    function used to schedule lr, wd, teacher momentum, teacher temp, last-layer lr.
    freeze steps at 0, then linear warmup start->base, then cosine base->final.
    (teacher_temp uses base==final so it is constant after warmup.)
    """
    if step < freeze:
        return 0.0
    if step < freeze + warmup:
        return start + (base - start) * (step - freeze) / max(1, warmup)
    progress = (step - freeze - warmup) / max(1, total - freeze - warmup)
    return final + 0.5 * (base - final) * (1.0 + math.cos(math.pi * progress))


class JWST_DINO(L.LightningModule):
    def __init__(
        self,
        # ── backbone (ViT size lives entirely here, no presets) ──
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        patch_size: int = 6,
        patch_stride: int = 3,
        in_chans: int = 1,
        num_register_tokens: int = 7,
        drop_path_rate: float = 0.2,
        layerscale_init: float = 1e-5,
        global_crops_size: int = 72,
        local_crops_size: int = 36,
        local_crops_number: int = 8,
        batch_size: int = 96,
        # ── projection head ──
        head_n_prototypes: int = 16384,
        head_bottleneck_dim: int = 192,
        head_nlayers: int = 3,
        head_hidden_dim: int = 768,
        # ── loss weights ──
        dino_loss_weight: float = 1.1,
        ibot_loss_weight: float = 1.0,
        koleo_loss_weight: float = 0.03,
        ibot_separate_head: bool = False,
        # ── teacher EMA + temperature ──
        momentum_teacher: float = 0.992,
        final_momentum_teacher: float = 1.0,
        teacher_temp: float = 0.07,
        warmup_teacher_temp: float = 0.04,
        warmup_teacher_temp_epochs: int = 10,
        student_temp: float = 0.1,
        # ── optimizer / schedule ──
        lr: float = 8.66e-5,
        min_lr: float = 5e-6,
        warmup_epochs: int = 5,
        weight_decay: float = 0.004,
        weight_decay_end: float = 0.04,
        clip_grad: float = 3.0,
        freeze_last_layer_epochs: int = 2,
        patch_embed_lr_mult: float = 0.2,
        layerwise_decay: float = 0.9,
        betas: List[float] = (0.9, 0.999),
        official_epoch_length: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        if ibot_separate_head:
            raise NotImplementedError(
                "ibot_separate_head=True is reserved but not implemented; iBOT shares the DINO head."
            )

        self.student_backbone = self._build_backbone(drop_path_rate)
        self.teacher_backbone = self._build_backbone(0.0)  # teacher: no stochastic depth
        self.student_dino_head = self._build_head()
        self.teacher_dino_head = self._build_head()

        # Teacher is an EMA of the student: start identical, never gets gradients.
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_dino_head.load_state_dict(self.student_dino_head.state_dict())
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_dino_head.parameters():
            p.requires_grad = False

        self.dino_loss = DINOLoss(head_n_prototypes, student_temp=student_temp)
        self.ibot_patch_loss = iBOTPatchLoss(head_n_prototypes, student_temp=student_temp)
        self.koleo_loss = KoLeoLoss()

    # ── builders ──────────────────────────────────────────────────────────────
    def _build_backbone(self, drop_path_rate: float) -> VisionTransformer:
        h = self.hparams
        return VisionTransformer(
            img_size=h.global_crops_size, patch_size=h.patch_size, patch_stride=h.patch_stride,
            in_chans=h.in_chans, embed_dim=h.embed_dim, depth=h.depth, num_heads=h.num_heads,
            mlp_ratio=h.mlp_ratio, num_register_tokens=h.num_register_tokens,
            drop_path_rate=drop_path_rate, layerscale_init=h.layerscale_init,
        )

    def train(self, mode: bool = True):
        """Keep the teacher in eval mode so its targets stay deterministic (no
        stochastic depth / dropout), matching DINOv2's train() override."""
        super().train(mode)
        self.teacher_backbone.eval()
        self.teacher_dino_head.eval()
        return self

    def _build_head(self) -> DINOHead:
        h = self.hparams
        return DINOHead(
            in_dim=h.embed_dim, out_dim=h.head_n_prototypes, hidden_dim=h.head_hidden_dim,
            bottleneck_dim=h.head_bottleneck_dim, nlayers=h.head_nlayers,
        )

    # ── losses shared by train / val ────────────────────────────────────────────
    def _compute_losses(self, batch: dict, teacher_temp: float, training: bool) -> dict:
        h = self.hparams
        global_crops = batch["collated_global_crops"]   # (2B, C, Hg, Wg)
        local_crops = batch["collated_local_crops"]     # (nL*B, C, Hl, Wl)
        masks = batch["collated_masks"]                 # (2B, N) bool
        masks_weight = batch["masks_weight"]            # (M,)
        B = global_crops.shape[0] // 2
        n_global, n_local = 2, h.local_crops_number

        # ── teacher (no grad): full global crops ──
        with torch.no_grad():
            t_out = self.teacher_backbone(global_crops)
            t_cls, t_patch = t_out["cls"], t_out["patch"]              # (2B,D), (2B,N,D)

            # DINO: cross-view pairing — reverse the two crops so crop0<->crop1.
            t_cls_rev = torch.cat([t_cls[B:], t_cls[:B]], dim=0)
            t_cls_head = self.teacher_dino_head(t_cls_rev)             # (2B, K)
            t_dino = self._center_softmax(self.dino_loss, t_cls_head, teacher_temp, training)

            # iBOT: teacher tokens at the masked positions (full-image targets).
            t_masked = self.teacher_dino_head(t_patch[masks])          # (M, K)
            t_ibot = self._center_softmax_ibot(t_masked, teacher_temp, training)

            if training:
                self.dino_loss.update_center(t_cls_head)
                self.ibot_patch_loss.update_center(t_masked.unsqueeze(0))

        # ── student: masked global crops + local crops ──
        s_glob = self.student_backbone(global_crops, masks=masks)
        s_glob_cls, s_glob_patch = s_glob["cls"], s_glob["patch"]
        s_loc_cls = self.student_backbone(local_crops)["cls"]

        s_glob_cls_head = self.student_dino_head(s_glob_cls)           # (2B, K)
        s_loc_cls_head = self.student_dino_head(s_loc_cls)             # (nL*B, K)
        s_masked_head = self.student_dino_head(s_glob_patch[masks])    # (M, K)

        # ── DINO loss (local + global), normalised exactly as DINOv2 ──
        n_local_terms = n_local * n_global
        n_global_terms = (n_global - 1) * n_global
        denom = n_global_terms + n_local_terms
        teacher_pair = list(t_dino.view(n_global, B, -1))             # [crop1, crop0]

        dino_local = self.dino_loss(s_loc_cls_head.chunk(n_local), teacher_pair) / denom
        dino_global = self.dino_loss([s_glob_cls_head], [t_dino]) * 2 / denom

        # ── iBOT loss (masked patch) ──
        ibot = self.ibot_patch_loss.forward_masked(
            s_masked_head, t_ibot, student_masks_flat=masks, masks_weight=masks_weight,
        ) * 2 * (1.0 / n_global)

        # ── KoLeo on the two student global CLS embeddings (pre-head) ──
        koleo = sum(self.koleo_loss(p) for p in s_glob_cls.chunk(n_global))

        total = (
            h.dino_loss_weight * (dino_global + dino_local)
            + h.ibot_loss_weight * ibot
            + h.koleo_loss_weight * koleo
        )
        return {
            "loss": total, "dino_global": dino_global, "dino_local": dino_local,
            "ibot": ibot, "koleo": koleo,
        }

    def _center_softmax(self, loss_obj, x, temp, training):
        """Teacher centering+sharpening. In training use the stateful EMA path; in
        validation read the current center without mutating it."""
        if training:
            return loss_obj.softmax_center_teacher(x, temp)
        return F.softmax((x - loss_obj.center) / temp, dim=-1)

    def _center_softmax_ibot(self, x, temp, training):
        x = x.unsqueeze(0)  # (1, M, K) — iBOT center is (1, 1, K)
        if training:
            return self.ibot_patch_loss.softmax_center_teacher(x, temp).squeeze(0)
        return F.softmax((x - self.ibot_patch_loss.center) / temp, dim=-1).squeeze(0)

    # ── training ────────────────────────────────────────────────────────────────
    def training_step(self, batch: dict, batch_idx: int = 0):
        B = batch["collated_global_crops"].shape[0] // 2
        opt = self.optimizers()
        lr, wd, mom, t_temp, last_lr = self._schedule_values()
        self._apply_lr_wd(opt, lr, wd, last_lr)

        out = self._compute_losses(batch, t_temp, training=True)

        opt.zero_grad()
        self.manual_backward(out["loss"])
        # Per-submodule grad clipping (backbone and head clipped independently).
        torch.nn.utils.clip_grad_norm_(self.student_backbone.parameters(), self.hparams.clip_grad)
        torch.nn.utils.clip_grad_norm_(self.student_dino_head.parameters(), self.hparams.clip_grad)
        opt.step()
        self._ema_update(mom)

        self.log_dict(
            {f"train_{k}": v for k, v in out.items()},
            prog_bar=True, on_step=True, on_epoch=False, batch_size=B,
        )
        # tokens_seen: config-invariant x-axis = cumulative student patch tokens (all crops).
        tokens_seen = self.global_step * B * self.trainer.world_size * self._tokens_per_image
        self.log_dict({"lr": lr, "wd": wd, "mom": mom, "teacher_temp": t_temp,
                       "tokens_seen": float(tokens_seen)},
                        on_step=True, on_epoch=False, batch_size=B)
        return out["loss"]

    def validation_step(self, batch: dict, batch_idx: int = 0, dataloader_idx: int = 0):
        B = batch["collated_global_crops"].shape[0] // 2
        _, _, _, t_temp, _ = self._schedule_values()
        out = self._compute_losses(batch, t_temp, training=False)

        # Per-survey val only: log val_<survey>_<k>. Each name appears under a single
        # dataloader_idx, so there is no cross-dataloader collision, and being logged in
        # validation_step (not on_validation_epoch_end) it lands in callback_metrics before
        # the EpochPrinter/PlotMetrics callbacks run. The combined "total" is derived from
        # these per-survey curves downstream, NOT logged here (see datamodule.setup note).
        surveys = getattr(self.trainer.datamodule, "val_surveys", None)
        if surveys is not None and dataloader_idx < len(surveys):
            tag = f"{surveys[dataloader_idx]}_"
        else:
            tag = ""  # single-survey / no datamodule: plain val_<k>
        self.log_dict({f"val_{tag}{k}": v for k, v in out.items()},
                        on_step=False, on_epoch=True, sync_dist=True,
                        batch_size=B, add_dataloader_idx=False)
        return out["loss"]

    @property
    def _effective_lr(self) -> float:
        return self.hparams.lr * math.sqrt(
            self.hparams.batch_size * self.trainer.world_size / 1024.0
        )

    @property
    def _tokens_per_image(self) -> int:
        """
        Computed Student patch token number per image across all crops (2 global + N local).
        This is used to standardize the training time across different setups (batch size, number of GPUs, etc.) by logging the number of tokens seen.
        """
        h = self.hparams
        sg = (h.global_crops_size - h.patch_size) // h.patch_stride + 1
        sl = (h.local_crops_size - h.patch_size) // h.patch_stride + 1
        return 2 * sg * sg + h.local_crops_number * sl * sl

    # ── schedules ────────────────────────────────────────────────────────────────
    def _schedule_values(self):
        h = self.hparams
        step = self.global_step
        total = max(int(self.trainer.estimated_stepping_batches), 1)
        epoch_len = h.official_epoch_length

        lr = cosine_sched(step, total, self._effective_lr, h.min_lr, warmup=h.warmup_epochs * epoch_len)
        wd = cosine_sched(step, total, h.weight_decay, h.weight_decay_end)
        mom = cosine_sched(step, total, h.momentum_teacher, h.final_momentum_teacher)
        t_temp = cosine_sched(
            step, total, h.teacher_temp, h.teacher_temp,
            warmup=h.warmup_teacher_temp_epochs * epoch_len, start=h.warmup_teacher_temp,
        )
        last_lr = 0.0 if step < h.freeze_last_layer_epochs * epoch_len else lr
        return lr, wd, mom, t_temp, last_lr

    def _apply_lr_wd(self, opt, lr, wd, last_lr):
        for g in opt.param_groups:
            g["weight_decay"] = wd * g["wd_multiplier"]
            g["lr"] = (last_lr if g["is_last_layer"] else lr) * g["lr_multiplier"]

    @torch.no_grad()
    def _ema_update(self, m: float):
        #loop over student/teacher backbone and head parameters in the same order, update teacher
        for ps, pt in zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)
        for ps, pt in zip(self.student_dino_head.parameters(), self.teacher_dino_head.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1 - m)

    # ── optimizer with layer-wise lr decay ──────────────────────────────────────
    def configure_optimizers(self):
        """Build one param group per parameter, tagged with lr/wd multipliers + last-layer flag.

        lr_multiplier = layerwise_decay ** (depth + 1 - layer_id); patch_embed gets
        an extra patch_embed_lr_mult; bias/norm/layerscale params get wd 0.
        """
        h = self.hparams
        n_blocks = h.depth
        groups = []
        named = list(self.student_backbone.named_parameters(prefix="backbone")) + \
            list(self.student_dino_head.named_parameters(prefix="dino_head"))
        for name, param in named:
            # skip frozen parameters
            if not param.requires_grad:
                continue

            layer_id = self._layer_id(name, n_blocks)
            lr_mult = h.layerwise_decay ** (n_blocks + 1 - layer_id)
            # update lr multipliers for patch_embed and last_layer wd multipliers for bias/norm params
            if "patch_embed" in name:
                lr_mult *= h.patch_embed_lr_mult
            wd_mult = 0.0 if (name.endswith(".bias") or "norm" in name or "gamma" in name) else 1.0
            
            groups.append({
                "params": [param], "name": name,
                "lr_multiplier": lr_mult, "wd_multiplier": wd_mult,
                "is_last_layer": "last_layer" in name,
            })
            
        return AdamW(groups, lr=self._effective_lr, betas=tuple(self.hparams.betas))

    @staticmethod
    def _layer_id(name: str, n_blocks: int) -> int:
        '''
        encode different layer types to assign a self-defined integer value
        will be usee in layer-wise learning rate decay
        '''
        
        if name.startswith("dino_head"):
            return n_blocks + 1
        if any(k in name for k in ("pos_embed", "cls_token", "register_tokens", "mask_token", "patch_embed")):
            return 0
        if ".blocks." in name:
            return int(name.split(".blocks.")[1].split(".")[0]) + 1
        return n_blocks + 1  # backbone.norm etc.

    # ── downstream / export ──────────────────────────────────────────────────────
    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Teacher CLS embedding for downstream tasks. images: (B, C, H, W)."""
        return self.teacher_backbone(images)["cls"]

    def export_teacher_backbone(self) -> dict:
        """State dict in the dinov2 'teacher checkpoint' layout used downstream."""
        return {"teacher": self.teacher_backbone.state_dict()}


_DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "jwst_dino.yaml")


def load_teacher_backbone(weights_path: str, device="cpu", config_path: str | None = None):
    """Load the frozen teacher ViT backbone for downstream eval (embeddings, probes).

    Accepts either checkpoint flavor and returns an eval-mode, frozen VisionTransformer
    whose expected input size is on ``.crop_size`` (call ``net(x)["cls"]`` for the
    embedding):

      *.ckpt — a Lightning ModelCheckpoint: rebuilt via ``JWST_DINO.load_from_checkpoint``
          (hyperparameters travel inside the ckpt, so no yaml is needed).
      *.pth  — the dinov2-style ``{"teacher": state_dict}`` dumped by
          ``export_teacher_backbone`` (backbone only, portable): the ViT is rebuilt from
          the training yaml at ``config_path`` (default ``jwst_dino.yaml``) and loaded.
    """
    if str(weights_path).endswith(".ckpt"):
        m = JWST_DINO.load_from_checkpoint(weights_path, map_location=device)
        net, crop = m.teacher_backbone, m.hparams.global_crops_size
    else:
        with open(config_path or _DEFAULT_CONFIG) as f:
            cfg = yaml.safe_load(f)
        mc, d = cfg["model"], cfg["data"]
        crop = d["global_crops_size"]
        net = VisionTransformer(
            img_size=crop, patch_size=d["patch_size"], patch_stride=d["patch_stride"],
            in_chans=mc["in_chans"], embed_dim=mc["embed_dim"], depth=mc["depth"],
            num_heads=mc["num_heads"], mlp_ratio=mc["mlp_ratio"],
            num_register_tokens=mc["num_register_tokens"], drop_path_rate=0.0,
            layerscale_init=mc["layerscale_init"],
        )
        sd = torch.load(weights_path, map_location="cpu")
        net.load_state_dict(sd.get("teacher", sd), strict=False)

    net = net.eval().to(device)
    for p in net.parameters():
        p.requires_grad = False
    net.crop_size = crop
    return net
