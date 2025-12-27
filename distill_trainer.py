# distill_trainer.py
import os
import time
from typing import Sequence, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== 通用工具 ===============
def get_logits_from_model_output(out):
    """从不同模型可能返回的结构里提取 logits [B,1,H,W]"""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for o in out:
            if isinstance(o, torch.Tensor) and o.dim() == 4:
                return o
        return out[0]
    if isinstance(out, dict):
        for k in ["logits", "masks", "mask", "out"]:
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
        for v in out.values():
            if isinstance(v, torch.Tensor):
                return v
    raise RuntimeError(f"Cannot extract logits from output type: {type(out)}")

def sigmoid_with_temp(logits, T=1.0):
    return torch.sigmoid(logits / T)

def kd_mse_prob(student_logits, teacher_logits, T=1.0):
    """输出蒸馏：MSE over probabilities with temperature."""
    ps = sigmoid_with_temp(student_logits, T)
    pt = sigmoid_with_temp(teacher_logits.detach(), T)
    return torch.mean((ps - pt) ** 2)

# =============== 中间特征蒸馏：基础损失 ===============
def loss_mse_feat(s, t):
    return F.mse_loss(s, t)

def loss_attention_transfer(s, t, eps=1e-6):
    """AT: 归一化空间注意力图的 L2"""
    def at_map(x):
        a = x.pow(2).mean(dim=1, keepdim=True)                  # [B,1,H,W]
        a = a / (a.norm(p=2, dim=(2,3), keepdim=True) + eps)
        return a
    return F.mse_loss(at_map(s), at_map(t))

def loss_channel_stat(s, t, eps=1e-5):
    """通道均值/标准差对齐"""
    def ch_stat(x):
        m = x.mean(dim=(2,3), keepdim=True)
        v = x.var(dim=(2,3), unbiased=False, keepdim=True).clamp_min(eps)
        return m, v.sqrt()
    ms, ss = ch_stat(s); mt, st = ch_stat(t)
    return F.l1_loss(ms, mt) + F.l1_loss(ss, st)

# =============== 中间特征蒸馏器 ===============
class KDFeatureDistiller(nn.Module):
    """
    从 teacher/student 的 ImageEncoder 取 backbone_fpn 做多层特征蒸馏。
    假设 encoder.forward(x) 返回:
        {
            "vision_features": ...,
            "vision_pos_enc": ...,
            "backbone_fpn": List[Tensor],  # 每层 (B, C_i, H_i, W_i)；已按 scalp 截断
        }
    """
    def __init__(
        self,
        teacher_encoder: nn.Module,
        student_encoder: nn.Module,
        teacher_channels: Sequence[int],
        student_channels: Sequence[int],
        distill_indices: Sequence[int],
        w_mse: float = 1.0,
        w_at: float = 250.0,
        w_ch: float = 1.0,
        align_to: str = "teacher",  # "teacher" or "student"
        use_1x1_adapters: bool = True,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert align_to in ("teacher", "student")
        self.teacher_encoder = teacher_encoder
        self.student_encoder = student_encoder
        self.distill_indices = list(distill_indices)
        self.w_mse = w_mse
        self.w_at = w_at
        self.w_ch = w_ch
        self.align_to = align_to

        # 通道适配器（student -> teacher）
        self.adapters = nn.ModuleDict()
        if use_1x1_adapters:
            for i, (cs, ct) in enumerate(zip(student_channels, teacher_channels)):
                self.adapters[str(i)] = nn.Conv2d(cs, ct, kernel_size=1, bias=False)
        else:
            for cs, ct in zip(student_channels, teacher_channels):
                assert cs == ct, "通道不一致且关闭了适配器，请开启 use_1x1_adapters=True"

        if device is not None:
            self.to(device)

        # 冻结教师编码器
        for p in self.teacher_encoder.parameters(): p.requires_grad = False
        self.teacher_encoder.eval()

    @torch.no_grad()
    def _enc_teacher(self, x: torch.Tensor) -> List[torch.Tensor]:
        out: Dict[str, Any] = self.teacher_encoder(x)
        return out["backbone_fpn"]

    def _enc_student(self, x: torch.Tensor) -> List[torch.Tensor]:
        out: Dict[str, Any] = self.student_encoder(x)
        return out["backbone_fpn"]

    def _align_pair(self, s: torch.Tensor, t: torch.Tensor, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 通道适配
        if str(idx) in self.adapters:
            s = self.adapters[str(idx)](s)
        # 尺寸对齐
        if self.align_to == "teacher":
            if s.shape[-2:] != t.shape[-2:]:
                s = F.adaptive_avg_pool2d(s, output_size=t.shape[-2:])
        else:
            if s.shape[-2:] != t.shape[-2:]:
                t = F.adaptive_avg_pool2d(t, output_size=s.shape[-2:])
        return s, t

    def forward(self, x_student: torch.Tensor, x_teacher: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        if x_teacher is None:
            x_teacher = x_student
        with torch.no_grad():
            t_feats_all = self._enc_teacher(x_teacher)
        s_feats_all = self._enc_student(x_student)

        kd_total = 0.0
        logs = {}
        for k, i in enumerate(self.distill_indices):
            s_feat = s_feats_all[i]
            t_feat = t_feats_all[i].detach()
            s_aligned, t_aligned = self._align_pair(s_feat, t_feat, idx=k)

            l_mse = loss_mse_feat(s_aligned, t_aligned)
            l_at  = loss_attention_transfer(s_aligned, t_aligned)
            l_ch  = loss_channel_stat(s_aligned, t_aligned)
            kd = self.w_mse * l_mse + self.w_at * l_at + self.w_ch * l_ch
            kd_total = kd_total + kd

            logs[f"kd_mse@{i}"] = float(l_mse.item())
            logs[f"kd_at@{i}"]  = float(l_at.item())
            logs[f"kd_ch@{i}"]  = float(l_ch.item())

        logs["kd_total"] = float(kd_total.item())
        return kd_total, logs

def build_kd_distiller_from_encoders(
    teacher_encoder: nn.Module,
    student_encoder: nn.Module,
    scalp_teacher: int = 1,
    scalp_student: int = 1,
    distill_last_n: int = 3,
    **kwargs,
) -> KDFeatureDistiller:
    """依据 encoder 内部 trunk.channel_list 自动取通道并构造蒸馏器。"""
    tch_list_all: Sequence[int] = teacher_encoder.trunk.channel_list
    stu_list_all: Sequence[int] = student_encoder.trunk.channel_list

    tch_list = tch_list_all[:-scalp_teacher] if scalp_teacher > 0 else list(tch_list_all)
    stu_list = stu_list_all[:-scalp_student] if scalp_student > 0 else list(stu_list_all)

    n = min(distill_last_n, len(tch_list), len(stu_list))
    distill_indices = list(range(-n, 0))
    tch_chs = [tch_list[i] for i in distill_indices]
    stu_chs = [stu_list[i] for i in distill_indices]

    return KDFeatureDistiller(
        teacher_encoder=teacher_encoder,
        student_encoder=student_encoder,
        teacher_channels=tch_chs,
        student_channels=stu_chs,
        distill_indices=distill_indices,
        **kwargs,
    )

# =============== Trainer ===============
class DistillTrainer:
    def __init__(
        self,
        model_dict: dict,
        student_name: str = "sam2_adapter_light",
        teacher_name: str = "sam2_adapter_tiny",
        student_ckpt: str | None = None,
        teacher_ckpt: str | None = None,
        criterion_bce: nn.Module | None = None,
        criterion_iou: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        kd_alpha: float = 1.0,     # 监督损失权重
        kd_beta_out: float = 1.0,  # 输出蒸馏权重
        kd_beta_feat: float = 1.0, # 中间特征蒸馏权重
        kd_T: float = 2.0,         # 输出蒸馏温度
        distill_last_n: int = 3,
        align_to: str = "teacher",
        use_1x1_adapters: bool = True,
        device: str = "cuda",
        student_size: int | None = None,   # 学生输入分辨率（None=使用数据集尺寸）
        teacher_size: int | None = None,   # 教师输入分辨率（None=与学生相同）
        save_dir: str = "./save/distill",
        logger=None,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.logger = logger

        # 构建模型
        self.student = model_dict[student_name]().to(self.device)
        self.teacher = model_dict[teacher_name]().to(self.device)

        # 可选：加载权重
        if student_ckpt is not None:
            self.student.load_state_dict(torch.load(student_ckpt, map_location=self.device))
        if teacher_ckpt is not None:
            self.teacher.load_state_dict(torch.load(teacher_ckpt, map_location=self.device))

        # 冻结教师
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        # 损失
        self.criterion_bce = criterion_bce or nn.BCEWithLogitsLoss().to(self.device)
        self.criterion_iou = criterion_iou or (lambda a,b: 0.0)  # 你可替换为自定义 IOU() 模块
        # 优化器调度器
        self.optimizer = optimizer or torch.optim.AdamW(self.student.parameters(), lr=1e-4, weight_decay=1e-2)
        self.scheduler = scheduler

        # KD 超参
        self.kd_alpha = kd_alpha
        self.kd_beta_out = kd_beta_out
        self.kd_beta_feat = kd_beta_feat
        self.kd_T = kd_T

        # 输入尺寸
        self.student_size = student_size
        self.teacher_size = teacher_size or student_size

        # 中间特征蒸馏器
        s_enc = getattr(self.student, "image_encoder")  # 按你的模型命名
        t_enc = getattr(self.teacher, "image_encoder")
        scalp_s = getattr(s_enc, "scalp", 1)
        scalp_t = getattr(t_enc, "scalp", 1)
        self.feat_distiller = build_kd_distiller_from_encoders(
            teacher_encoder=t_enc,
            student_encoder=s_enc,
            scalp_teacher=scalp_t,
            scalp_student=scalp_s,
            distill_last_n=distill_last_n,
            w_mse=1.0, w_at=250.0, w_ch=1.0,
            align_to=align_to,
            use_1x1_adapters=use_1x1_adapters,
            device=self.device,
        )

        self.best_val_loss = float("inf")

    def _resize_if_needed(self, x: torch.Tensor, size: int | None) -> torch.Tensor:
        if size is None or x.shape[-1] == size:
            return x
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)

    def train_one_epoch(self, train_loader):
        self.student.train()
        meter = []
        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                input, target, *rest = batch
            else:
                input, target = batch, None
                rest = []

            input = input.to(self.device)
            target = target.to(self.device).unsqueeze(1).float()

            # 学生前向
            student_inp = self._resize_if_needed(input, self.student_size)
            student_out = self.student(student_inp, True)
            student_logits = get_logits_from_model_output(student_out)

            # 教师前向（no grad）
            with torch.no_grad():
                teacher_inp = self._resize_if_needed(input, self.teacher_size)
                teacher_out = self.teacher(teacher_inp, True)
                teacher_logits = get_logits_from_model_output(teacher_out)
                # 对齐到学生输出尺寸
                if teacher_logits.shape[-2:] != student_logits.shape[-2:]:
                    teacher_logits = F.interpolate(teacher_logits, size=student_logits.shape[-2:], mode="bilinear", align_corners=False)

            # 监督损失
            sup_loss = self.criterion_bce(student_logits, F.interpolate(target, size=student_logits.shape[-2:], mode="nearest"))
            iou_loss = self.criterion_iou(student_logits, F.interpolate(target, size=student_logits.shape[-2:], mode="nearest"))
            supervised = sup_loss + (iou_loss if not isinstance(iou_loss, (int, float)) else 0.0)

            # 输出蒸馏
            distill_out = kd_mse_prob(student_logits, teacher_logits, T=self.kd_T)

            # 中间特征蒸馏
            kd_feat, kd_logs = self.feat_distiller(x_student=student_inp, x_teacher=teacher_inp)

            # 总损失
            loss = self.kd_alpha * supervised + self.kd_beta_out * distill_out + self.kd_beta_feat * kd_feat

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
            self.optimizer.step()

            meter.append({
                "loss": float(loss.item()),
                "sup": float(supervised.item()),
                "kd_out": float(distill_out.item()),
                "kd_feat": float(kd_feat.item())
            })

        if self.scheduler is not None:
            self.scheduler.step()

        # 简要日志
        avg_loss = sum(m["loss"] for m in meter) / max(len(meter), 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader):
        self.student.eval()
        meter = []
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                input, target, *rest = batch
            else:
                input, target = batch, None

            input = input.to(self.device)
            target = target.to(self.device).unsqueeze(1).float()

            student_inp = self._resize_if_needed(input, self.student_size)
            out = self.student(student_inp)
            logits = get_logits_from_model_output(out)

            sup_loss = self.criterion_bce(logits, F.interpolate(target, size=logits.shape[-2:], mode="nearest"))
            iou_loss = self.criterion_iou(logits, F.interpolate(target, size=logits.shape[-2:], mode="nearest"))
            total = sup_loss + (iou_loss if not isinstance(iou_loss, (int, float)) else 0.0)

            meter.append(float(total.item()))

        return sum(meter) / max(len(meter), 1)

    def fit(self, train_loader, val_loader=None, epochs: int = 100, test_freq: int = 10, save_name: str = "student_best.pth"):
        print("==> KD training...")
        for epoch in range(1, epochs+1):
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader)
            t1 = time.time()
            print(f"epoch {epoch}, train_loss={train_loss:.4f}, time={t1-t0:.2f}s")

            if val_loader is not None:
                val_loss = self.validate(val_loader)
                print(f"epoch {epoch}, val_loss={val_loss:.4f}")
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.student.state_dict(), os.path.join(self.save_dir, save_name))
                    print(f"  ✓ save best to {os.path.join(self.save_dir, save_name)}")

            if (val_loader is not None) and (epoch % test_freq == 0):
                print(f"--- eval at epoch {epoch}: best_val_loss={self.best_val_loss:.4f} ---")
