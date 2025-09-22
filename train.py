# train.py  — U-Net 单类分割（压痕区域），每轮独立保存验证可视化

import os
os.environ.setdefault("A_DISABLE_VERSION_CHECK", "1")  # 关掉 albumentations 的升级提示

import cv2
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Windows/CPU 更稳：限制线程，避免卡顿
torch.set_num_threads(4)

from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# =========================
# 数据集
# =========================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# =========================
# 数据集
# =========================
class VickersDataset(Dataset):
    """
    单类语义分割数据集：用于识别维氏压痕（前景=压痕，背景=其他）

    参数 ----
    image_paths : List[str]
        原始图像的绝对/相对路径列表（通常来自 data/images/*.jpg 等）。
        注意：本类会根据 image_paths 推断掩膜目录在 ../masks 下，并按同名 .png 寻找掩膜。
    img_size : int
        网络输入的目标尺寸。图像会先等比缩放到最长边不超过该值，再用常数填充到正方形(img_size×img_size)。
    aug : bool
        是否开启数据增强。训练集应设为 True（提升泛化），验证/测试集应设为 False（保持评估稳定）。

    目录与命名约定
    --------------
    假设原图在 data/images/xxx.jpg，对应的掩膜应为 data/masks/xxx.png。
    掩膜为单通道图，前景像素 > 0 视为 1，背景 = 0。
    """
    def __init__(self, image_paths: List[str], img_size: int = 512, aug: bool = False):
        # 基本参数检查
        assert len(image_paths) > 0, "空的 image_paths"
        self.image_paths = image_paths

        # 推断掩膜目录：images 的上级目录 + /masks
        # e.g. data/images/xxx.jpg -> data/masks/xxx.png
        self.mask_dir = str(Path(self.image_paths[0]).parent.parent / "masks")
        self.img_size = img_size

        # ====== 构建 Albumentations 变换流水线 ======
        if aug:
            # 训练集：随机增强（旋转/翻转/亮度对比度/模糊/噪声等）
            # 说明：Albumentations 对 mask 默认使用“最近邻插值”，可以保持掩膜边缘不被插值模糊
            self.tf = A.Compose([
                # 1) 等比缩放：将最长边缩放到不超过 img_size（不改变纵横比，避免拉伸）
                #    对图像使用双线性插值；对掩膜会用最近邻插值（由 Albumentations 自动处理）
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),

                # 2) 填充到正方形：若某一边短于 img_size，则在右/下补常数值（黑色）到 img_size×img_size
                #    统一网络输入尺寸，同时不破坏原始纵横比
                A.PadIfNeeded(min_height=img_size, min_width=img_size,
                              border_mode=cv2.BORDER_CONSTANT),

                # 3) 离散方向增强（80% 概率触发；三者等概率择一）：
                #    - 水平翻转
                #    - 垂直翻转
                #    - 90°的整数倍旋转（0/90/180/270）
                #    目的：让模型对“方向”更不敏感（维氏压痕旋转角度不固定）
                A.OneOf([
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                    A.RandomRotate90(p=1.0),
                ], p=0.8),

                # 4) 任意角度旋转（±180°，60% 概率）：进一步提升旋转不变性
                #    副作用：会引入黑角/黑边（但我们已在上一步 pad 成正方形，能减少裁切）
                A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.6),

                # 5) 成像风格扰动（80% 概率触发；三者等概率择一）：
                #    - RandomBrightnessContrast：亮度/对比度随机变动（模拟光照差异）
                #    - CLAHE：自适应直方图均衡（提升局部对比度，增强纹理）
                #    - GaussianBlur：轻微高斯模糊（模拟轻微失焦/抖动）
                #    注意：若任务对“边界清晰度”极其敏感，可降低模糊的权重或移除
                A.OneOf([
                    A.RandomBrightnessContrast(p=1.0),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.8),

                # 6) 高斯噪声（30% 概率）：模拟传感器噪声/高 ISO 场景
                #    仅作用于图像，不作用于掩膜
                A.GaussNoise(p=0.3),

                # 7) 标准化：使用 ImageNet 的均值和方差（与 encoder_weights="imagenet" 对齐）
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),

                # 8) 转张量：HWC(np.float32) -> CHW(torch.float32)
                ToTensorV2(),
            ])
        else:
            # 验证/测试集：仅做确定性的几何预处理 + 标准化（不做随机增强，保证评估可比）
            self.tf = A.Compose([
                # 等比缩放到最长边 ≤ img_size
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR),

                # 填充到正方形 img_size×img_size
                A.PadIfNeeded(min_height=img_size, min_width=img_size,
                              border_mode=cv2.BORDER_CONSTANT),

                # 标准化（与训练一致）
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),

                # 转张量
                ToTensorV2(),
            ])

    # 返回样本数量：用于 len(dataset) 与 DataLoader 的长度计算
    def __len__(self):
        return len(self.image_paths)

    # 读取 RGB 图像（原始存储是 BGR，需转成 RGB 供网络使用）
    def _read_image(self, p: str):
        """
        参数  ----
        p : str
            图像路径
        返回 ----
        np.ndarray (H, W, 3), dtype=uint8, RGB 排列
        """
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        # OpenCV 默认读出来是 BGR，这里转为 RGB
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 读取掩膜：单通道；>0 为前景，=0 为背景
    def _read_mask(self, img_path: str):
        """
        参数 ----
        img_path : str
            与该图像同名的掩膜将从 ../masks/name.png 读取
        返回 ----
        np.ndarray (H, W), dtype=uint8, 取值 {0,1}
        """
        name = Path(img_path).stem
        mp = os.path.join(self.mask_dir, f"{name}.png")
        m = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(mp)
        # 若掩膜不小心存成了 3 通道图，取第一通道使用
        if m.ndim == 3:
            m = m[:, :, 0]
        # 二值化：>0 视为前景(1)，否则为背景(0)
        m = (m > 0).astype(np.uint8)
        return m

    # 随机/顺序索引获取一个样本：返回 (图像张量, 掩膜张量, 文件名stem)
    def __getitem__(self, i: int):
        """
        参数 ----
        i : int
            索引
        返回  ----
        x : torch.FloatTensor, 形状 [3, H, W]
            归一化后的图像张量
        y : torch.FloatTensor, 形状 [1, H, W]
            二值掩膜张量（0/1），在 ToTensor 后额外 unsqueeze(0)
        name : str
            文件名（不含扩展名），用于可视化/保存结果时命名
        """
        # 1) 读原图 & 掩膜
        ip = self.image_paths[i]
        img = self._read_image(ip)
        msk = self._read_mask(ip)

        # 2) 应用变换（Albumentations 会对 image 和 mask 做“同步”的几何变换）
        tf = self.tf(image=img, mask=msk)

        # 3) 取出张量：image -> [3,H,W], mask -> [H,W]
        x = tf["image"].float()
        # 掩膜扩一维成 [1,H,W]，并转 float（后续损失使用 BCEWithLogits，需要 float）
        y = tf["mask"].unsqueeze(0).float()

        # 4) 返回三元组（训练循环里还会返回 name 用于保存可视化）
        return x, y, Path(ip).stem


# =========================
# 工具函数
# =========================
# 设置 Python / NumPy / PyTorch 的随机种子，便于复现实验结果。
def set_seed(seed: int = 42) -> None:
    """
    参数 ----
    seed : int
        随机种子。相同的 seed 可以保证：
        - Python 内置随机库（random）
        - NumPy 随机数
        - PyTorch CPU 与 CUDA 的随机数（包括权重初始化、DataLoader shuffle 等）
      在相同代码、相同环境下尽量生成一致的结果。
      注意：若涉及到 CuDNN 非确定性算子或多进程 DataLoader，
      仍可能存在极小的不可复现性；此处已满足大多数场景。
    """
    # Python 随机库
    random.seed(seed)
    # NumPy 随机库
    np.random.seed(seed)
    # PyTorch CPU 随机数（权重初始化、某些算子等）
    torch.manual_seed(seed)
    # PyTorch CUDA 随机数（若有 GPU）
    torch.cuda.manual_seed_all(seed)

# 计算批量（batch）的 Dice 系数（F1 的一种形式），反映预测与真实掩膜的重叠程度。
# 计算流程：对概率图进行 0.5 阈值化 -> 计算每张图的 Dice -> 对 batch 取平均。
def dice_coef(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """
    参数 ----
    prob : torch.Tensor
        模型输出的**概率图**（经过 sigmoid 的结果），形状为 [N, 1, H, W]，
        取值范围 [0, 1]。若传入的是 logits，请先对外部做 torch.sigmoid。
    target : torch.Tensor
        真实掩膜（二值），形状为 [N, 1, H, W]，取值通常是 {0, 1} 的 float/bool。
    eps : float
        数值稳定项，避免分母为 0 导致的 NaN。当预测与目标均为全零时尤为重要。

    返回 ----
    float
        Dice 系数在 [0, 1]，越大越好。返回的是“对 batch 求平均后”的标量。
    """
    # 将概率图阈值化为二值预测（>0.5 视为前景 1），保持计算简单直观
    pred = (prob > 0.5).float()

    # 计算交集（intersection）与并集（union 的 Dice 版本是 pred.sum + target.sum）
    # dim=(1,2,3) 表示在通道与空间维度上求和，仅保留 batch 维（N）
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    # Dice = (2 * 交集) / (并集)
    # 加上 eps 防止 0/0，最后对 batch 取平均并转成 Python float
    return ((2 * inter + eps) / (union + eps)).mean().item()

# 计算批量（batch）的 IoU（Jaccard 指数），反映预测与真实掩膜的交并比。
#  计算流程：对概率图进行 0.5 阈值化 -> 计算每张图的 IoU -> 对 batch 取平均。
def iou_coef(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    """
    参数 ----
    prob : torch.Tensor
        模型输出的**概率图**（经过 sigmoid 的结果），形状为 [N, 1, H, W]。
    target : torch.Tensor
        真实掩膜（二值），形状为 [N, 1, H, W]。
    eps : float
        数值稳定项，避免分母为 0。

    返回 ----
    float
        IoU 系数在 [0, 1]，越大越好。返回的是“对 batch 求平均后”的标量。
    """
    # 将概率图阈值化为二值预测
    pred = (prob > 0.5).float()

    # 交集：pred AND target；并集：pred OR target
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter

    # IoU = 交集 / 并集
    return ((inter + eps) / (union + eps)).mean().item()

# 将一个 batch 内“每张图片”的可视化结果单独保存为一张“四联图”：
# [原图 | GT 掩膜 | 预测掩膜 | 叠加可视化]，便于肉眼检查模型效果。
def save_individual_visuals(
    x: torch.Tensor,
    y: torch.Tensor,
    pr: torch.Tensor,
    names: list,
    out_dir: Path
) -> None:
    """
    参数 ----
    x : torch.Tensor
        归一化后的图像张量，形状为 [N, 3, H, W]，范围大致在标准化后的分布（非 0~255）。
        注意：这里会进行“反标准化”恢复到 0~255 的可视化强度。
    y : torch.Tensor
        真实掩膜张量，形状为 [N, 1, H, W]，取值通常为 {0, 1} 的 float。
    pr : torch.Tensor
        预测概率张量，形状为 [N, 1, H, W]，取值范围 [0, 1]（一般为 sigmoid 后的结果）。
        这里用于可视化时会线性缩放到 0~255，但**不会二值化**，
        便于观察置信度（越白 = 概率越高）。
    names : list[str]
        与 batch 对应的文件名列表（无扩展名），用于输出文件命名。
    out_dir : Path
        输出目录；不存在则自动创建。

    生成的文件 --------
    out_dir / "{name}.jpg"
        横向拼接的四联图（BGR 色彩空间，OpenCV 默认），分辨率为 H×(4W)。
    """
    # 确保输出目录存在
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) 反标准化并还原到 0~255 可视范围（与训练时 Normalize 的 mean/std 对应） ----
    # x: [N, 3, H, W] -> [N, H, W, 3]
    x_np = x.permute(0, 2, 3, 1).cpu().numpy()
    # 逐通道反标准化：img = (img * std + mean) * 255
    # 这里的 mean/std 必须与训练时 A.Normalize 中的数值保持一致
    x_np = (x_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255.0
    # 裁剪到 [0,255] 并转为 uint8
    x_np = np.clip(x_np, 0, 255).astype(np.uint8)

    # ---- 2) 掩膜/预测图转换到可视范围 ----
    # y: [N, 1, H, W] -> [N, H, W]，映射到 0/255（白=前景，黑=背景）
    y_np = (y.squeeze(1).cpu().numpy() * 255).astype(np.uint8)
    # pr: [N, 1, H, W] -> [N, H, W]，概率 [0,1] 映射到 [0,255]（越白=置信度越高）
    pr_np = (pr.squeeze(1).cpu().numpy() * 255).astype(np.uint8)

    # ---- 3) 逐张保存 ----
    for i, name in enumerate(names):
        # OpenCV 使用 BGR，这里将 RGB -> BGR 以保证颜色正常
        rgb = x_np[i][:, :, ::-1]  # RGB -> BGR
        gt  = y_np[i]              # [H, W]  0/255
        pd  = pr_np[i]             # [H, W]  0~255

        # 叠加图：将预测区域着色（BGR: 橙色 (0,140,255)），alpha 控制透明度
        color = np.zeros_like(rgb)                # [H, W, 3], 全 0
        color[pd > 127] = (0, 140, 255)          # 概率>0.5 的区域上色
        vis = cv2.addWeighted(rgb, 1.0, color, 0.35, 0.0)

        # 将灰度的 gt/pd 扩成 3 通道，便于与彩色图横向拼接
        gt_vis = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
        pd_vis = cv2.cvtColor(pd, cv2.COLOR_GRAY2BGR)

        # 横向拼接：[原图 | GT 掩膜 | 预测掩膜 | 叠加]
        canvas = np.hstack([rgb, gt_vis, pd_vis, vis])

        # 写出到磁盘（JPEG 默认质量 95；若目录很大可调低质量减少体积）
        cv2.imwrite(str(out_dir / f"{name}.jpg"), canvas)



# =========================
# 训练 / 验证
# =========================
def build_model(encoder: str = "resnet34", weights: str = "imagenet"):
    """
    构建 U-Net 语义分割模型（单类输出）

    参数 ----
    encoder : str
        用作 U-Net 编码器（backbone）的网络名称。常见选项：'resnet18'/'resnet34'/'resnet50' 等。
        - 编码器负责提取多层级特征，放在 U-Net 的下采样路径上。
    weights : str
        是否加载 ImageNet 预训练权重（推荐 'imagenet'），可以加速收敛，提高泛化。

    返回 ----
    model : torch.nn.Module
        已构建的 U-Net 模型实例。
    """
    model = smp.Unet(
        encoder_name=encoder,        # U-Net 编码器类型（骨干网络）
        encoder_weights=weights,     # 是否加载预训练参数（通常设为 'imagenet'）
        in_channels=3,                # 输入通道数（RGB 彩图 = 3）
        classes=1,                    # 输出类别数（单通道二分类分割）
        activation=None               # 不加激活（直接输出 logits，后面损失用 BCEWithLogitsLoss）
    )
    return model

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn_bce,
    loss_fn_dice,
    device,
    scaler=None
):
    """
    训练模型一个 epoch

    参数 ----
    model : nn.Module
        要训练的模型。
    loader : DataLoader
        训练集 DataLoader，按 batch 产出 (x, y, name)。
    optimizer : torch.optim.Optimizer
        优化器（如 AdamW / SGD）。
    loss_fn_bce : nn.Module
        BCEWithLogitsLoss 实例，用于二分类交叉熵。
    loss_fn_dice : nn.Module
        DiceLoss 实例，用于重叠区域损失。
    device : str
        "cuda" 或 "cpu"，模型和数据都会被转到这个设备上。
    scaler : torch.amp.GradScaler, optional
        混合精度训练的梯度缩放器（仅在 CUDA 有效；CPU 无效）

    返回 ---
    float
        本 epoch 的平均训练损失。
    """
    model.train()             # 开启训练模式（启用 Dropout / BN 等）
    t_loss, count = 0.0, 0

    # tqdm 进度条（ncols=100：固定宽度，避免换行乱）
    pbar = tqdm(loader, desc="Train", ncols=100, leave=True)

    # 是否使用混合精度（仅 GPU）
    use_amp = (device == "cuda")

    # === 遍历训练集的每个 batch ===
    for x, y, _ in pbar:
        # 将 batch 数据转到目标设备
        x, y = x.to(device), y.to(device)

        # 清空优化器梯度（set_to_none=True 会减少显存碎片）
        optimizer.zero_grad(set_to_none=True)

        # 开启自动混合精度上下文（autocast）
        with torch.amp.autocast(
            device_type=("cuda" if use_amp else "cpu"),
            dtype=(torch.float16 if use_amp else torch.bfloat16),
            enabled=use_amp
        ):
            logits = model(x)                          # 前向传播
            # 总损失 = BCE + Dice
            loss = loss_fn_bce(logits, y) + loss_fn_dice(logits, y)

        # === 反向传播 & 参数更新 ===
        if scaler is not None and use_amp:
            # 混合精度：缩放 loss -> 反传 -> 更新 -> 恢复缩放
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 普通精度训练
            loss.backward()
            optimizer.step()

        # 累积损失（loss.item() 是当前 batch 的平均损失）
        t_loss += loss.item() * x.size(0)
        count  += x.size(0)

        # tqdm 进度条实时显示当前平均 loss
        pbar.set_postfix(loss=f"{t_loss/max(1,count):.4f}")

    # 返回整个 epoch 的平均训练损失
    return t_loss / max(1, count)

@torch.no_grad() # 临时关闭梯度计算，让其中的代码块在执行时不会构建计算图、不会记录梯度、不会占用显存保存中间梯度结果。
def validate(
    model,
    loader,
    loss_fn_bce,
    loss_fn_dice,
    device,
    out_vis_dir: Path = None,
    save_every_image: bool = True
):
    """
    在验证集上评估模型（不更新参数）

    参数 ----
    model : nn.Module
        要评估的模型。
    loader : DataLoader
        验证集 DataLoader。
    loss_fn_bce : nn.Module
        BCEWithLogitsLoss。
    loss_fn_dice : nn.Module
        DiceLoss。
    device : str
        "cuda" 或 "cpu"。
    out_vis_dir : Path, optional
        若不为 None，将把每张验证图的可视化结果保存到此目录。
    save_every_image : bool
        是否保存验证集**每一张图**的可视化。
        如果数据很多，可设为 False 只保留一部分，节省时间。

    返回 ----
    (val_loss, mean_dice, mean_iou) : (float, float, float)
        本 epoch 验证集的平均损失、Dice、IoU。
    """
    model.eval()   # 评估模式（冻结 Dropout/BN）
    v_loss, count = 0.0, 0
    dices, ious = [], []

    pbar = tqdm(loader, desc="Val  ", ncols=100, leave=True)

    # 创建输出目录（保存预测可视化）
    if out_vis_dir is not None:
        out_vis_dir.mkdir(parents=True, exist_ok=True)

    # === 遍历验证集 ===
    for x, y, names in pbar:
        x, y = x.to(device), y.to(device)

        # 前向传播
        logits = model(x)

        # 损失：BCE + Dice
        loss = loss_fn_bce(logits, y) + loss_fn_dice(logits, y)
        v_loss += loss.item() * x.size(0)
        count += x.size(0)

        # 预测概率（sigmoid 把 logits -> [0,1] 概率）
        prob = torch.sigmoid(logits)

        # 计算每个 batch 的 Dice/IoU 并收集
        dices.append(dice_coef(prob, y))
        ious.append(iou_coef(prob, y))

        # 保存预测可视化
        if out_vis_dir is not None and save_every_image:
            save_individual_visuals(x, y, prob, names, out_vis_dir)

    # 返回三项指标：平均验证损失 / 平均 Dice / 平均 IoU
    return v_loss / max(1, count), float(np.mean(dices)), float(np.mean(ious))



# =========================
# 主入口
# =========================
def run(cfg: Dict[str, Any]):
    """
    训练主流程（从读数据 -> 构建模型 -> 训练/验证多轮 -> 保存最优/最后权重 -> 尝试导出ONNX）

    参数 ----
    cfg : Dict[str, Any]
        训练配置字典。示例见文件底部 RECOMMENDED_CFG。
        关键项说明：
        - data: 数据根目录（必须包含 images/ 与 masks/）
        - img_size: 统一输入的正方形尺寸（如 384/512）
        - epochs: 训练轮数
        - batch: 批大小
        - lr: 初始学习率
        - encoder: U-Net 编码器骨干（如 'resnet34'）
        - encoder_weights: 'imagenet' 或 None
        - val_ratio: 验证集比例（0~1）
        - out: 输出目录（保存权重/日志/可视化）
        - seed: 随机种子
        - dump_all_val: 是否每轮保存验证集所有图片的可视化
        - dump_all_train: 是否每轮保存训练集所有图片的可视化（很慢，默认 False）
        - early_stop_patience: 早停轮数阈值，None 表示不启用
    """

    # 1) 固定随机种子，保证划分与初始权重可复现
    set_seed(cfg.get("seed", 42))

    # 2) 扫描数据：读取 data/images 下所有图像路径
    data_dir = Path(cfg["data"])
    img_dir  = data_dir / "images"
    assert img_dir.exists(), f"{img_dir} 不存在"
    # 仅保留允许的扩展名（IMG_EXTS 在文件顶部定义）
    all_imgs = sorted([str(p) for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS])
    assert len(all_imgs) > 0, f"{img_dir} 下没有图像文件"

    # 3) 训练/验证划分（按 seed 随机打乱后取前 n_val 做验证）
    r = random.Random(cfg.get("seed", 42))
    imgs = all_imgs[:]; r.shuffle(imgs)
    n_val = max(1, int(len(imgs) * cfg.get("val_ratio", 0.1)))
    val_imgs = imgs[:n_val]
    train_imgs = imgs[n_val:]

    # 4) 构建 Dataset
    #    - 训练集 aug=True：启用随机增强（提高泛化）
    #    - 验证集 aug=False：不做随机增强（评估稳定可比）
    train_ds = VickersDataset(train_imgs, img_size=cfg["img_size"], aug=True)
    val_ds   = VickersDataset(val_imgs,   img_size=cfg["img_size"], aug=False)

    # 5) 构建 DataLoader（Windows/CPU 友好设置）
    #    - num_workers=0：避免 Windows 多进程 DataLoader 卡住
    #    - pin_memory=False：CPU 环境没必要；若 CUDA 可将其设 True 微幅提速
    train_dl = DataLoader(train_ds, batch_size=cfg["batch"], shuffle=True,
                          num_workers=0, pin_memory=False, persistent_workers=False)
    val_dl   = DataLoader(val_ds,   batch_size=cfg["batch"], shuffle=False,
                          num_workers=0, pin_memory=False, persistent_workers=False)

    # 6) 设备选择：优先使用 CUDA；否则用 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 7) 构建 U-Net 模型（单通道输出，用 BCEWithLogits + Dice）
    model = build_model(cfg["encoder"], cfg["encoder_weights"]).to(device)

    # 8) 定义损失函数：
    #    - BCEWithLogitsLoss：二分类像素交叉熵（内部自带 sigmoid）
    #    - DiceLoss(binary)：重叠区域损失，提升对不均衡前景的鲁棒性
    loss_bce  = nn.BCEWithLogitsLoss()
    loss_dice = smp.losses.DiceLoss(mode="binary")

    # 9) 优化器 + 学习率调度器
    #    - AdamW：权重衰减 decoupled，泛化更稳
    #    - CosineAnnealingLR：从初始 lr 余弦退火到接近 0；T_max=epochs 即每个 epoch 调整一次
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    # 10) AMP 梯度缩放器（仅 CUDA 有效；CPU 自动 disabled）
    from torch.amp import GradScaler
    scaler = GradScaler('cuda', enabled=(device=="cuda"))

    # 11) 准备输出目录（保存可视化/权重/训练曲线）
    out_dir = Path(cfg["out"])
    (out_dir / "val_vis").mkdir(parents=True, exist_ok=True)

    # 12) 训练前信息打印（使用 tqdm.write 避免破坏进度条）
    tqdm.write(f"[INFO] Train: {len(train_ds)}  Val: {len(val_ds)}  Device: {device}")
    tqdm.write(f"[INFO] Save dir: {out_dir.resolve()}")

    # 13) 指标与日志缓存
    best_dice = -1.0      # 跟踪最佳验证 Dice
    history = []          # 保存每轮指标到 JSON

    # 14) 早停（Early Stopping）控制变量
    patience = cfg.get("early_stop_patience", None)  # None 表示不启用早停
    no_improve = 0                                  # 连续未提升计数

    # ========================= 训练主循环 =========================
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # 14.1) 训练一轮：返回该轮平均训练损失
        train_loss = train_one_epoch(
            model, train_dl, optimizer, loss_bce, loss_dice, device, scaler
        )

        # 14.2) 验证一轮：保存每轮的可视化到独立子目录（ep_XXX）
        ep_vis_dir = (out_dir / "val_vis" / f"ep_{ep:03d}")
        val_loss, val_dice, val_iou = validate(
            model, val_dl, loss_bce, loss_dice, device,
            out_vis_dir=ep_vis_dir,
            save_every_image=cfg.get("dump_all_val", True)  # True=保存验证集所有图片
        )

        # 14.3) 学习率调度器步进（余弦退火）
        scheduler.step()

        # 14.4) 记录与打印本轮指标
        rec = {
            "epoch": ep,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_dice": float(val_dice),
            "val_iou": float(val_iou),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "time": round(time.time()-t0, 2)  # 单轮耗时（秒）
        }
        history.append(rec)
        print(
            f"Ep {ep:03d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"Dice {val_dice:.4f} | IoU {val_iou:.4f} | lr {rec['lr']:.2e} | {rec['time']}s"
        )

        # 14.5) 保存权重
        #      - best.pth：若本轮 val_dice 提升，则覆盖保存（用于最终推理/部署）
        #      - last.pth：每轮覆盖保存（便于恢复最后状态）
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), out_dir / "best.pth")
            print(f"  ✓ Saved best.pth (Dice={best_dice:.4f})")
            no_improve = 0  # 重置未提升计数
        else:
            no_improve += 1
            if patience is not None:
                print(f"  ↺ no improvement for {no_improve}/{patience} epochs")

        torch.save(model.state_dict(), out_dir / "last.pth")

        # 14.6) 写入训练曲线日志（JSON），便于后续绘图或分析
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        # 14.7) （可选）训练集可视化：逐张保存（非常慢，默认关闭）
        if cfg.get("dump_all_train", False):
            model.eval()
            with torch.no_grad():
                pbar = tqdm(train_dl, desc="DumpTrainVis", ncols=100, leave=True)
                ep_train_vis_dir = (out_dir / "train_vis" / f"ep_{ep:03d}")
                ep_train_vis_dir.mkdir(parents=True, exist_ok=True)
                for x, y, names in pbar:
                    x, y = x.to(device), y.to(device)
                    prob = torch.sigmoid(model(x))
                    save_individual_visuals(x, y, prob, names, ep_train_vis_dir)

        # 14.8) 早停检查：若连续 no_improve 轮数达到 patience，则停止训练
        if patience is not None and no_improve >= patience:
            print(f"[EARLY STOP] val_dice has not improved for {patience} epochs. Stop at epoch {ep}.")
            break

    # ========================= 训练结束，尝试导出 ONNX =========================
    try:
        model.eval()
        # 构造一个虚拟输入（N=1, C=3, H=W=img_size），指定 device 保持一致
        dummy = torch.randn(1, 3, cfg["img_size"], cfg["img_size"], device=device)
        onnx_path = str(out_dir / "unet.onnx")
        # 导出
        exported = torch.onnx.dynamo_export(model, dummy)  # 这一步就已跟踪模型
        exported.save(onnx_path)  # 直接保存 .onnx
        print(f"[INFO] Exported ONNX to {onnx_path}")
    except Exception as e:
        # 未安装 onnx/onnxruntime 或算子不支持时会报错；训练本身不受影响
        print(f"[WARN] ONNX 导出失败：{e}")



# =========================
# 用推荐参数直接启动（CPU 友好）
# =========================
if __name__ == "__main__":
    # =============================
    # 默认推荐配置（RECOMMENDED_CFG）
    # =============================
    RECOMMENDED_CFG = {
        # 数据集根目录，需包含：
        #   data/images/xxx.png   → 原图
        #   data/masks/xxx.png    → 对应掩码（0=背景，>0=前景）
        "data": "data",

        # 输入图像会被缩放/填充为 (img_size, img_size)
        # - 越大精度可能更高，但内存占用也更高
        # - 384 在 CPU 上比较轻量，512 更适合 GPU
        "img_size": 512,

        # 最大训练轮数（每轮=完整遍历一次训练集）
        # - 实际可能提前停止（如果启用了早停）
        "epochs": 500,

        # 每个 batch 的图像数
        # - CPU 建议设小一点（1~2），GPU 可设大（8~16）
        "batch": 8,

        # 初始学习率
        # - 太大容易震荡不收敛，太小收敛很慢
        "lr": 5e-5,

        # U-Net 的编码器（骨干网络）类型
        # - 可选 'resnet18' / 'resnet34' / 'resnet50' / 'efficientnet-b0' 等
        "encoder": "resnet34",

        # 是否加载 ImageNet 预训练权重
        # - 推荐 "imagenet"，能提升收敛速度和精度
        "encoder_weights": "imagenet",

        # 验证集比例（0~1）
        # - 用于自动随机划分验证集
        # - 0.1 表示 10% 验证、90% 训练
        "val_ratio": 0.1,

        # 所有训练输出（模型权重/日志/可视化图）保存到此目录下
        "out": "runs/unet_r34_512",

        # 随机种子（控制数据划分、权重初始化等）
        # - 固定种子保证每次运行结果一致，方便对比
        "seed": 42,

        # 是否在每一轮后把**验证集的每张图**都保存可视化结果
        # - True：便于观察训练进展
        # - 缺点：比较耗时和占用磁盘空间
        "dump_all_val": True,

        # 是否也保存**训练集**的可视化（通常关闭）
        # - True：会非常慢，几乎不需要
        # - 一般只在 debug 阶段用
        "dump_all_train": False,

        # 早停机制（Early Stopping）
        # - None：关闭早停，始终训练满 epochs 轮
        # - 例如设 10：如果 val_dice 连续 10 轮无提升则提前停止
        "early_stop_patience": None
    }

    # 启动训练流程
    run(RECOMMENDED_CFG)

