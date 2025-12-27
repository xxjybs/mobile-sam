import torch
import torch.nn as nn
from torch.nn import functional as F


def D_L2(f1, f2):
    return torch.norm(f1-f2)


def Normal(f):
    mean = torch.mean(f, dim=(2, 3), keepdim=True)
    std = torch.std(f, dim=(2, 3), keepdim=True)
    return (f-mean)/std

class NFD_loss_after_conv1x1(nn.Module):
    def __init__(self, t_channel, s_channel):
        """
        这里默认教师网络的channel大于学生网络，注意这个前提
        :param t_channel:
        :param s_channel:
        """
        super(NFD_loss_after_conv1x1, self).__init__()
        self.t_channel = t_channel
        self.s_channel = s_channel
        # self.conv1x1 = nn.Conv2d(s_channel, t_channel, 1)

    def forward(self, f_t, f_s):
        t_N, t_C, t_W, t_H = f_t.shape
        s_N, s_C, s_W, s_H = f_s.shape
        if t_W != s_W or t_H != s_H:
            f_t = F.interpolate(f_t, size=(s_W, s_H), mode='bilinear', align_corners=True)
        # if self.t_channel != self.s_channel:
        #     f_s = self.conv1x1(f_s)
        f_t.detach()
        if f_s.shape != f_t.shape:
            print(f_s.shape, f_t.shape)
        return D_L2(f_t, f_s) / (s_W * s_H)

# --------------------- KD（logits 蒸馏） ---------------------
def kd_loss_logits(student_logit, teacher_logit, T=2.0):
    # 对齐分辨率
    if student_logit.shape[-2:] != teacher_logit.shape[-2:]:
        teacher_logit = F.interpolate(teacher_logit, size=student_logit.shape[-2:], mode="bilinear", align_corners=False)

    # 二分类时把 1ch 扩为 (bg, fg) 2ch，避免数值不稳
    def to_two_channel(logit):
        return torch.cat([-logit, logit], dim=1) if logit.shape[1] == 1 else logit

    s = to_two_channel(student_logit) / T
    t = to_two_channel(teacher_logit).detach() / T

    log_p_s = F.log_softmax(s, dim=1)
    p_t    = F.softmax(t, dim=1)

    # reduction='none' -> 按通道求和，再对 HW 和 B 取平均，最后乘 T^2
    kl_per_pixel = F.kl_div(log_p_s, p_t, reduction='none').sum(dim=1)  # sum over classes
    loss = kl_per_pixel.mean() * (T * T)  # mean over B,H,W
    return loss


def pixel_loss(output, target):
    """
    pixel loss
    """
    return F.mse_loss(output, target)

def kl_pixel_loss(output_t, output_s, alpha=0.5, temperature=2):
    """
    kd loss = 0.5 * pixel loss + 0.5 * kl loss
    """
    _, _, t_W, t_H = output_t.shape
    _, _, W, H = output_s.shape
    if t_W == W and t_H == H:
        output_t = output_t
    else:
        output_t = F.interpolate(output_t, (W, H), mode='bilinear', align_corners=True)
    output_t.detach()
    soft_output_s = F.softmax(output_s / temperature, dim=1)
    soft_output_t = F.softmax(output_t / temperature, dim=1)
    kd_loss = F.kl_div(soft_output_s.log(), soft_output_t, reduction='batchmean') * (temperature ** 2)
    pixel_loss_value = pixel_loss(output_s, output_t)
    kd_loss = alpha * kd_loss + (1 - alpha) * pixel_loss_value
    return kd_loss / (W * H)



if __name__ == '__main__':
    f_t = torch.randn(16, 4, 512, 512)
    f_s = torch.randn(16, 4, 224, 224)
    f_t_1 = torch.randn(16, 512, 512, 512)
    f_s_1 = torch.randn(16, 224, 224, 224)
    loss = kl_pixel_loss(f_t, f_s)
    y = NFD_loss_after_conv1x1(512, 224)
    loss1 = y(f_t_1, f_s_1)
    print(loss)
    print(loss1)
