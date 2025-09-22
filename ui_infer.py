# -*- coding: utf-8 -*-
"""
ui_infer.py
=======================================
功能概述：
- 使用 U-Net（ONNX 或 PyTorch）对维氏压痕进行语义分割，得到概率掩膜；
- 在概率掩膜基础上进行 OpenCV 几何后处理，支持同时识别多个压痕；
- 可视化为 2×3 网格：
  第一行：原图 | 掩膜(灰度) | 半透明覆盖
  第二行：几何结果@原图 | 几何结果@二值 | 几何结果@覆盖
- 底部日志打印耗时与检测结果（每个压痕的 d1/d2/mean、面积、中心等）。

模型加载优先级（自动选择）：
1) 目录存在 unet.onnx ——> ONNX Runtime（优先 CUDAExecutionProvider，再回退 CPU）
2) 否则若存在 last.pth/best.pth ——> PyTorch + segmentation_models_pytorch.Unet(resnet34)
3) 两者都没有 ——> 不做推理，仅显示原图并提示

注意：
- DEFAULT_IMG_SIZE 必须与训练时一致（例如 512），否则会影响精度或模型输入不匹配。
- 本脚本采用 letterbox（等比缩放+补边）使输入变为正方形；推理后再用 unletterbox 还原回原图尺寸。
"""

from pathlib import Path
import time
import numpy as np
import cv2
from PySide6 import QtCore, QtGui, QtWidgets

# ==============================
# 可调参数（按需修改）
# ==============================
DEFAULT_MODEL_DIR = Path("runs/unet_r34_512")   # 模型目录：放 unet.onnx 或 best/last.pth
DEFAULT_IMG_SIZE  = 512                         # 模型输入正方形尺寸（需与训练一致）
IMAGENET_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # 归一化均值
IMAGENET_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # 归一化方差

# 后处理默认参数（可在 on_open() 里统一修改）
BIN_THRESH       = 0.50    # 概率二值化阈值（分割保守可降到 0.45；粘连多可升到 0.55）
MIN_AREA_FRAC    = 0.0008  # 最小连通域面积占比（相对整图面积），噪点多/分辨率高可调大到 0.001~0.003
MORPH_KERNEL     = 3       # 形态学核大小（奇数：3/5/7），纹理噪点多可用 5
OPEN_ITER        = 1       # 开运算迭代次数（去毛刺/小噪点）
CLOSE_ITER       = 1       # 闭运算迭代次数（连通缺口）


# ==============================
# UI 小组件：可等比缩放的 QLabel
# ==============================
class ScaledLabel(QtWidgets.QLabel):
    """
    一个保持长宽比的图片显示控件：
    - setPixmap() 时记录原始 QPixmap；
    - resizeEvent() 时做等比缩放；
    - 使用 SmoothTransformation，缩放效果更平滑。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orig_pm: QtGui.QPixmap | None = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)

    def setPixmap(self, pm: QtGui.QPixmap) -> None:  # type: ignore[override]
        self._orig_pm = pm
        self._apply_scaled()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(e)
        if self._orig_pm is not None:
            self._apply_scaled()

    def _apply_scaled(self):
        if self._orig_pm is None:
            return
        scaled = self._orig_pm.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


# ==============================
# 图像预/后处理工具函数
# ==============================
def letterbox_square(img_bgr: np.ndarray, size: int, pad_value=0):
    """
    将任意 H×W 图像等比缩放，并补边到 (size, size)。
    仅下采样（避免放大导致虚化），记录缩放比例与补边信息，便于推理后还原。

    返回：
        img_pad: (size, size, 3) BGR 图像
        scale  : 缩放比例（原始 -> 缩放后）
        pad    : (top, bottom, left, right) 补边像素
    """
    h, w = img_bgr.shape[:2]
    scale = min(size / max(h, w), 1.0)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_rs = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left

    img_pad = cv2.copyMakeBorder(
        img_rs, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
    )
    return img_pad, scale, (top, bottom, left, right)


def unletterbox(mask_sq: np.ndarray, scale: float, pad: tuple, orig_hw: tuple):
    """
    与 letterbox 相反：将 (size, size) 的单通道掩膜/概率图
    去除补边，并按 1/scale 缩放回原图尺寸。
    """
    top, bottom, left, right = pad
    size = mask_sq.shape[0]
    crop = mask_sq[top:size - bottom, left:size - right]
    oh, ow = orig_hw
    if crop.shape[:2] == (oh, ow):
        return crop
    return cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)


def to_qpixmap_from_bgr(img_bgr: np.ndarray) -> QtGui.QPixmap:
    """
    OpenCV BGR(numpy) -> Qt QPixmap
    注意 Qt 使用 RGB 排列，因此先做 BGR->RGB。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    qimg = QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def make_overlay(img_bgr: np.ndarray, mask01: np.ndarray, color=(0, 0, 255), alpha=0.35):
    """
    在 mask>0.5 的区域上叠加半透明颜色层（默认红色）。
    alpha 越大覆盖越明显。
    """
    base = img_bgr.copy()
    overlay = np.zeros_like(base, dtype=np.uint8)
    overlay[mask01 > 0.5] = color
    return cv2.addWeighted(base, 1.0, overlay, alpha, 0)


# ==============================
# 多目标几何后处理（基于 minAreaRect）
# ==============================
def postprocess_minarearect_multi(
    img_bgr: np.ndarray,
    prob01: np.ndarray,
    bin_thresh: float = BIN_THRESH,
    min_area_frac: float = MIN_AREA_FRAC,
    morph_kernel: int = MORPH_KERNEL,
    open_iter: int = OPEN_ITER,
    close_iter: int = CLOSE_ITER,
):
    """
    概率图 -> 二值 -> 形态学清理 -> 连通域筛选(>=最小面积) -> 对每个连通域做 minAreaRect
    通过 boxPoints 得到四个角点，**不依赖顶点排序**；
    用“**长度最大的两条且不共点**”的线段作为对角线，避免将边误判为对角线。

    返回：
      clean_bin: 清理后的二值图 (uint8, 0/255)
      detections: list[dict]，按面积从大到小排序：
          {
            "label": int 连通域标签,
            "area": int 连通域面积（像素）,
            "box": np.ndarray(4,2) 四角坐标（int32）,
            "center": (cx, cy) 旋转矩形中心（float）,
            "d1": float, "d2": float, "d_mean": float   # 两条对角线长度及其均值（像素）
          }
    """
    h, w = prob01.shape[:2]
    min_area = max(200, int(min_area_frac * h * w))  # 像素最小面积，至少 200

    # 1) 概率二值化（可根据样本调整 bin_thresh）
    mask = (prob01 >= bin_thresh).astype(np.uint8) * 255

    # 2) 形态学清理（开：去毛刺；闭：补小洞）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)

    # 3) 连通域筛选（保留所有 >= min_area 的区域）
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    keep_labels = []
    for i in range(1, num_labels):  # 0 是背景
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == i] = 255
            keep_labels.append((i, area))

    detections = []
    if not keep_labels:
        return clean, detections

    # 4) 对每个保留连通域做 minAreaRect
    for i, area in keep_labels:
        # 为了得到该连通域的轮廓，构造该连通域的单独二值图
        mask_i = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        rect = cv2.minAreaRect(cnt)           # ((cx,cy), (w,h), angle)
        box  = cv2.boxPoints(rect).astype(np.int32)  # 4个角点（顺序非 TL/TR/BR/BL）

        # 自动找两条“最长且不共点”的线段作为对角线（对四点所有两两组合求距离）
        pairs = []
        for a in range(4):
            for b in range(a + 1, 4):
                d = float(np.linalg.norm(box[a] - box[b]))
                pairs.append((d, a, b))
        pairs.sort(reverse=True, key=lambda x: x[0])
        # 最长的一条
        _, i1, j1 = pairs[0]
        # 第二条必须与第一条不共享端点（剩余两个点自然就是另一条对角线）
        rest = [k for k in range(4) if k not in (i1, j1)]
        i2, j2 = rest[0], rest[1]

        d1 = float(np.linalg.norm(box[i1] - box[j1]))
        d2 = float(np.linalg.norm(box[i2] - box[j2]))
        detections.append({
            "label": i,
            "area": area,
            "box": box,
            "center": (float(rect[0][0]), float(rect[0][1])),
            "d1": d1, "d2": d2, "d_mean": 0.5 * (d1 + d2),
        })

    # 5) 统一排序（默认按面积从大到小；也可换成按 d_mean 排序）
    detections.sort(key=lambda x: x["area"], reverse=True)
    return clean, detections


def draw_detections_on_three(
    img_bgr: np.ndarray,
    clean_bin: np.ndarray,
    overlay_bgr: np.ndarray,
    detections: list,
):
    """
    将多个检测结果同时画到三张图上：
      - vis_o：原图
      - vis_b：二值图（3 通道 BGR）
      - vis_v：半透明覆盖图
    对每个目标画：旋转矩形外框 + 两条对角线 + 文本标签（编号与 mean）。

    返回：
      vis_o, vis_b, vis_v
    """
    # 颜色集合（循环使用，提升可读性）
    palette = [
        (0, 255, 0),     # 绿
        (255, 0, 0),     # 蓝
        (0, 255, 255),   # 黄
        (255, 0, 255),   # 品红
        (0, 165, 255),   # 橙
        (255, 255, 0),   # 青
        (147, 20, 255),  # 紫
        (50, 205, 50),   # 黄绿
    ]
    diag_color = (0, 0, 255)  # 对角线统一红色

    vis_o = img_bgr.copy()
    vis_b = cv2.cvtColor(clean_bin, cv2.COLOR_GRAY2BGR)
    vis_v = overlay_bgr.copy()

    for idx, det in enumerate(detections, start=1):
        box = det["box"].astype(np.int32)
        color_box = palette[(idx - 1) % len(palette)]

        # 找到两条对角线（再次按“最长且不共点”的规则，稳妥）
        pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                d = float(np.linalg.norm(box[i] - box[j]))
                pairs.append((d, i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        _, i1, j1 = pairs[0]
        rest = [k for k in range(4) if k not in (i1, j1)]
        i2, j2 = rest[0], rest[1]

        # 在三张图上统一绘制
        for canvas in (vis_o, vis_b, vis_v):
            cv2.polylines(canvas, [box.reshape(-1, 1, 2)], True, color_box, 2)
            cv2.line(canvas, tuple(box[i1]), tuple(box[j1]), diag_color, 2)
            cv2.line(canvas, tuple(box[i2]), tuple(box[j2]), diag_color, 2)

            # 标注编号与 mean（在中心点右上方）
            cx, cy = int(det["center"][0]), int(det["center"][1])
            label_txt = f"#{idx} mean={det['d_mean']:.1f}px"
            cv2.putText(canvas, label_txt, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2, cv2.LINE_AA)

    return vis_o, vis_b, vis_v


# ==============================
# 模型封装（ONNX / PyTorch 自适应）
# ==============================
class Segmenter:
    """
    UNet 推理封装：
    - 自动选择 ONNX 或 PyTorch（SMP.Unet(resnet34)）；
    - 输入使用 letterbox 到 (S,S)，输出再 unletterbox 还原；
    - 返回与原图同尺寸的概率图（float32, [0..1]）。
    """
    def __init__(self, model_dir: Path, img_size: int):
        self.model_dir = Path(model_dir)
        self.img_size = img_size
        self.kind = None     # "onnx" / "pth" / None
        self._ok = False

        # 约定的模型文件名（也可自行扩展成自动搜索）
        self.onnx_path = self.model_dir / "unet.onnx"
        self.pth_path  = self.model_dir / "last.pth"
        if not self.pth_path.exists():
            self.pth_path = self.model_dir / "best.pth"

        # 1) 优先尝试 ONNX
        if self.onnx_path.exists():
            try:
                import onnxruntime as ort
                self.sess = ort.InferenceSession(
                    str(self.onnx_path),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                )
                self.input_name = self.sess.get_inputs()[0].name
                self.kind = "onnx"
                self._ok = True
            except Exception as e:
                self.sess = None
                print("[WARN] ONNX 加载失败:", e)

        # 2) 回退到 PyTorch（segmentation_models_pytorch）
        if not self._ok and self.pth_path.exists():
            try:
                import torch
                import segmentation_models_pytorch as smp
                self.torch = torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                # 注意：下面这些参数需与训练保持一致
                self.model = smp.Unet(
                    encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=1, activation=None
                ).to(self.device)
                sd = torch.load(self.pth_path, map_location=self.device)
                self.model.load_state_dict(sd, strict=True)
                self.model.eval()
                self.kind = "pth"
                self._ok = True
            except Exception as e:
                self.model = None
                print("[WARN] PTH 加载失败:", e)

    def ok(self) -> bool:
        return self._ok

    def info(self) -> str:
        if not self._ok:
            return "未加载模型"
        return f"ONNX: {self.onnx_path.name}" if self.kind == "onnx" else f"PTH: {self.pth_path.name} (resnet34)"

    def preprocess(self, img_bgr: np.ndarray):
        """
        预处理：
        - letterbox 到 (S,S)；
        - BGR->RGB，归一化到 [0,1]，再做 (x-mean)/std；
        - HWC -> CHW，扩展 batch 维。
        """
        h0, w0 = img_bgr.shape[:2]
        sq, scale, pad = letterbox_square(img_bgr, self.img_size, pad_value=0)

        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(rgb, (2, 0, 1))              # HWC->CHW
        inp = np.expand_dims(chw, 0).astype(np.float32) # [1,3,S,S]
        return inp, (scale, pad, (h0, w0))

    def infer(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        前向推理：
        返回与原图同尺寸的概率图 prob（float32，[0..1]）。
        """
        if not self._ok:
            raise RuntimeError("模型未加载")
        inp, meta = self.preprocess(img_bgr)
        scale, pad, orig_hw = meta

        if self.kind == "onnx":
            out = self.sess.run(None, {self.input_name: inp})[0]  # [1,1,S,S]
            prob_sq = 1.0 / (1.0 + np.exp(-out))                 # logits -> sigmoid
            prob_sq = prob_sq[0, 0]                              # (S,S)
        else:
            torch = self.torch
            x = torch.from_numpy(inp).to(self.device, non_blocking=True)
            with torch.inference_mode():
                logits = self.model(x)                           # [1,1,S,S]
                prob_sq = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()

        # 还原回原图尺寸（去补边 + 缩放回去）
        prob = unletterbox(prob_sq.astype(np.float32), scale, pad, orig_hw)
        prob = np.clip(prob, 0.0, 1.0)
        return prob


# ==============================
# 主窗口（2×3 网格）
# ==============================
class MainWindow(QtWidgets.QMainWindow):
    """
    UI 布局：
      第一行：原图 / 掩膜(灰度) / 半透明覆盖
      第二行：几何结果@原图 / 几何结果@二值 / 几何结果@覆盖
    底部：日志（打印时耗与多目标检测结果）
    """
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, img_size=DEFAULT_IMG_SIZE):
        super().__init__()
        self.setWindowTitle("维氏压痕识别 — 多目标 | UNet + OpenCV(minAreaRect)")
        self.resize(1680, 980)

        # 模型
        self.segmenter = Segmenter(model_dir, img_size)

        # 顶部：打开按钮 + 模型信息
        self.open_btn = QtWidgets.QPushButton("打开图片…")
        self.open_btn.clicked.connect(self.on_open)
        self.model_label = QtWidgets.QLabel(self.segmenter.info())
        self.model_label.setStyleSheet("color:#777;")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.open_btn)
        top.addSpacing(12)
        top.addWidget(self.model_label)
        top.addStretch(1)

        # 6 个视图（两行三列）
        self.view_orig   = ScaledLabel()  # 原图
        self.view_mask   = ScaledLabel()  # 掩膜(灰度)
        self.view_ovl    = ScaledLabel()  # 半透明覆盖
        self.view_geom_o = ScaledLabel()  # 几何结果@原图
        self.view_geom_b = ScaledLabel()  # 几何结果@二值
        self.view_geom_v = ScaledLabel()  # 几何结果@覆盖

        # 统一设置基础属性
        for v in (self.view_orig, self.view_mask, self.view_ovl,
                  self.view_geom_o, self.view_geom_b, self.view_geom_v):
            v.setMinimumSize(220, 220)
            v.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        # 网格布局
        grid = QtWidgets.QGridLayout()
        # 第一行
        grid.addWidget(self._titled("原图", self.view_orig),         0, 0)
        grid.addWidget(self._titled("掩膜(灰度)", self.view_mask),   0, 1)
        grid.addWidget(self._titled("半透明覆盖", self.view_ovl),    0, 2)
        # 第二行
        grid.addWidget(self._titled("几何结果@原图", self.view_geom_o), 1, 0)
        grid.addWidget(self._titled("几何结果@二值", self.view_geom_b), 1, 1)
        grid.addWidget(self._titled("几何结果@覆盖", self.view_geom_v), 1, 2)

        for c in range(3):
            grid.setColumnStretch(c, 1)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # 底部日志
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMinimumHeight(160)
        self.log.setStyleSheet("font-family: Consolas, Menlo, monospace; font-size:12px;")

        # 中央布局
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addLayout(top)
        lay.addLayout(grid, 1)
        lay.addWidget(self.log)
        self.setCentralWidget(central)

        if not self.segmenter.ok():
            self.append_log(f"[WARN] 未找到模型：{model_dir}")
            self.append_log("请将 unet.onnx 或 best/last.pth 放入该目录。")

    def _titled(self, title, widget):
        """给子控件加标题边框"""
        box = QtWidgets.QGroupBox(title)
        l = QtWidgets.QVBoxLayout(box)
        l.addWidget(widget)
        return box

    def append_log(self, text: str):
        """带时间戳的日志输出"""
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {text}")

    @QtCore.Slot()
    def on_open(self):
        """文件对话框 -> 读取图片 -> 推理 -> 生成 6 张可视化图 -> 日志"""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return

        t0 = time.perf_counter()
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.append_log(f"读取失败：{path}")
            return

        # 第一行：原图
        self.view_orig.setPixmap(to_qpixmap_from_bgr(img))

        if not self.segmenter.ok():
            self.append_log("模型未加载，仅显示原图。")
            return

        # 推理（计时）
        t1 = time.perf_counter()
        prob = self.segmenter.infer(img)  # (H,W) float32 [0..1]
        t2 = time.perf_counter()

        # 第一行：掩膜灰度
        mask_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        self.view_mask.setPixmap(to_qpixmap_from_bgr(mask_bgr))

        # 第一行：半透明覆盖
        overlay = make_overlay(img, prob, color=(0, 0, 255), alpha=0.35)
        self.view_ovl.setPixmap(to_qpixmap_from_bgr(overlay))

        # 多目标后处理（核心）
        clean, detections = postprocess_minarearect_multi(
            img, prob,
            bin_thresh=BIN_THRESH,
            min_area_frac=MIN_AREA_FRAC,
            morph_kernel=MORPH_KERNEL,
            open_iter=OPEN_ITER,
            close_iter=CLOSE_ITER,
        )

        # 第二行：将所有检测结果画到三张图
        vis_o, vis_b, vis_v = draw_detections_on_three(img, clean, overlay, detections)
        self.view_geom_o.setPixmap(to_qpixmap_from_bgr(vis_o))
        self.view_geom_b.setPixmap(to_qpixmap_from_bgr(vis_b))
        self.view_geom_v.setPixmap(to_qpixmap_from_bgr(vis_v))

        # 日志：时耗
        self.append_log(f"打开：{Path(path).name}  尺寸={img.shape[1]}×{img.shape[0]}")
        self.append_log(
            f"预处理+I/O: {(t1 - t0)*1e3:.1f} ms | 推理: {(t2 - t1)*1e3:.1f} ms | 总计: {(t2 - t0)*1e3:.1f} ms"
        )
        self.append_log(f"模型：{self.segmenter.info()}  img_size={self.segmenter.img_size}")

        # 日志：检测结果
        if not detections:
            self.append_log("后处理：未检测到压痕。可调 bin_thresh/min_area_frac/morph_kernel。")
        else:
            self.append_log(f"检测到 {len(detections)} 个压痕（按面积降序）：")
            for i, det in enumerate(detections, 1):
                cx, cy = det["center"]
                self.append_log(
                    f"  #{i} label={det['label']} | area={det['area']} | "
                    f"d1={det['d1']:.1f}px, d2={det['d2']:.1f}px, mean={det['d_mean']:.1f}px | "
                    f"center=({cx:.1f},{cy:.1f})"
                )


# ==============================
# 入口
# ==============================
def main():
    app = QtWidgets.QApplication([])
    w = MainWindow(model_dir=DEFAULT_MODEL_DIR, img_size=DEFAULT_IMG_SIZE)
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
