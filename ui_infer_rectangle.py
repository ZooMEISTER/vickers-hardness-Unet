# -*- coding: utf-8 -*-
"""
ui_infer.py
=======================================
功能概述（带超详细注释版）：
- 使用 U-Net（ONNX 或 PyTorch）对维氏压痕进行语义分割，得到概率掩膜；
- 在概率掩膜基础上进行 OpenCV 几何后处理（minAreaRect），支持**同时识别多个压痕**；
- 将可视化结果渲染成 2×3 网格（共 6 张图）：
  第一行：原图 | 掩膜(灰度) | 半透明覆盖
  第二行：几何结果@原图 | 几何结果@二值 | 几何结果@覆盖
- 底部日志打印耗时与检测结果（每个压痕的 d1/d2/mean、面积、中心等）。
- 交互增强：点击任意一张图，会弹出一个可缩放/可拖拽的大图预览窗口。

模型加载优先级（自动选择）：
1) 若目录存在 unet.onnx ——> 使用 ONNX Runtime 推理（优先 CUDAExecutionProvider，再回退 CPU）
2) 否则若存在 last.pth 或 best.pth ——> 使用 PyTorch + segmentation_models_pytorch.Unet(resnet34)
3) 若两者都没有 ——> 不做推理，仅显示原图并日志提示缺模型

注意事项：
- DEFAULT_IMG_SIZE 必须与训练时一致（例如 512），否则会导致精度下降或推理输入不匹配。
- 本脚本采用 letterbox（等比缩放 + 补边）将任意长宽比图像适配为正方形输入；推理输出再通过
  unletterbox 还原回原图尺寸（去补边 + 按 1/scale 等比缩放）。

依赖：
- PySide6（Qt for Python），OpenCV（cv2），numpy
- 可选：onnxruntime，torch，segmentation_models_pytorch
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

# 后处理默认参数（可在 on_open() 中统一修改）
BIN_THRESH       = 0.50    # 概率二值化阈值。分割保守可降到 0.45；粘连多可升到 0.55
MIN_AREA_FRAC    = 0.0008  # 最小连通域面积占整图比例。噪点多/分辨率高可调到 0.001~0.003
MORPH_KERNEL     = 3       # 形态学核大小（奇数：3/5/7），纹理噪点多可用 5
OPEN_ITER        = 1       # 开运算迭代次数（去毛刺/小噪点）
CLOSE_ITER       = 1       # 闭运算迭代次数（连通缺口）


# =============================================================================
# 大图预览弹窗：基于 QGraphicsView，实现滚轮缩放 + 拖拽平移 + 双击复位
# =============================================================================
class ZoomImageDialog(QtWidgets.QDialog):
    """
    简洁的图片预览弹窗：
    - 使用 QGraphicsView/QGraphicsScene/QGraphicsPixmapItem 展示 QPixmap；
    - 支持滚轮缩放（以鼠标所在位置为锚点，更自然）；
    - 左键拖拽平移；双击视窗复位至“适配窗口”的缩放；
    - 使用非模态（.show()）打开，可同时查看多张图；若想阻塞用 .exec() 亦可。
    """
    def __init__(self, pixmap: QtGui.QPixmap, title: str = "预览", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1024, 720)  # 初始尺寸
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)

        # 1) 视图与场景初始化
        self._view = _GraphicsView(self)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._item = QtWidgets.QGraphicsPixmapItem(pixmap)  # 实际承载位图的 Item
        self._scene.addItem(self._item)
        self._view.setScene(self._scene)

        # 2) 简单的无边距布局：全窗铺满视图
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._view)

        # 3) 等窗口渲染后做一次 fitInView，使图片初始完整显示
        QtCore.QTimer.singleShot(0, self._view.fitInViewSmooth)

    def setPixmap(self, pm: QtGui.QPixmap):
        """
        若你想在弹窗中切换展示的图片，可调用此方法。
        会自动重新做一次 fitInView 适配。
        """
        self._item.setPixmap(pm)
        self._view.fitInViewSmooth()


class _GraphicsView(QtWidgets.QGraphicsView):
    """
    QGraphicsView 的轻定制版本：
    - 启用抗锯齿与平滑位图变换，保证缩放质量；
    - 滚轮缩放：以鼠标为锚点（AnchorUnderMouse）；
    - 拖拽平移：ScrollHandDrag 模式；
    - 双击：复位 transform 并自适应视图；
    - 限制最小/最大缩放比例，防止过度缩放损坏体验。
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        # 渲染优化
        self.setRenderHints(QtGui.QPainter.Antialiasing |
                            QtGui.QPainter.SmoothPixmapTransform |
                            QtGui.QPainter.TextAntialiasing)
        # 拖动模式：按住鼠标左键拖拽即可平移
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        # 变换/缩放的锚点：以鼠标为中心，更符合直觉
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)

        # 视觉上更干净：不显示滚动条（靠拖拽平移）
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # 约束缩放范围（当前 transform 的 m11/m22 为尺度近似值）
        self._min_scale = 0.05
        self._max_scale = 50.0

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        """
        鼠标滚轮缩放：
        - event.angleDelta().y() > 0 代表向上滚（放大），反之缩小；
        - 每次缩放比例为 1.25 或 1/1.25；
        - 基于当前 transform 的 m11 值做范围约束，避免无限缩放。
        """
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1/1.25

        cur_m = self.transform()
        cur_scale = cur_m.m11()  # 以 x 方向的缩放系数做估计（假设等比缩放）
        new_scale = cur_scale * factor
        if new_scale < self._min_scale and factor < 1:
            factor = self._min_scale / max(cur_scale, 1e-9)
        elif new_scale > self._max_scale and factor > 1:
            factor = self._max_scale / max(cur_scale, 1e-9)

        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        """
        双击：重置所有缩放/位移，并做一个“适配视图”的缩放（保持纵横比）。
        """
        self.resetTransform()
        self.fitInViewSmooth()

    def fitInViewSmooth(self):
        """
        将场景整体“缩放居中适配”到视图中（保持纵横比），作为初始与复位行为。
        """
        scene_rect = self.sceneRect()
        if scene_rect.isNull():
            return
        self.fitInView(scene_rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)


# =============================================================================
# UI 小组件：可等比缩放的可点击 QLabel（发射 clicked(QPixmap) 信号）
# =============================================================================
class ScaledLabel(QtWidgets.QLabel):
    """
    一个保持长宽比的图像显示控件 + 点击预览：
    - setPixmap()：记录“原始 QPixmap”（未缩放版本），保存到 self._orig_pm；
    - resizeEvent()：对 _orig_pm 做等比缩放绘制，保证不被拉伸；
    - 鼠标左键点击：如果已有 _orig_pm，则发射 clicked(QPixmap) 信号；
    - 通过设置 Cursor 为“手型”，给用户点击提示。
    """
    clicked = QtCore.Signal(QtGui.QPixmap)  # 对外暴露的“点击”信号，传递原始 QPixmap

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orig_pm: QtGui.QPixmap | None = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def setPixmap(self, pm: QtGui.QPixmap) -> None:  # type: ignore[override]
        """
        外部设置图像的唯一入口：
        - 保存未缩放的原图 QPixmap（用于点击预览与再次缩放）；
        - 触发一次等比缩放绘制以适配当前控件大小。
        """
        self._orig_pm = pm
        self._apply_scaled()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        """
        当控件尺寸变化时，基于原始 QPixmap 重新生成等比缩放版本并显示。
        """
        super().resizeEvent(e)
        if self._orig_pm is not None:
            self._apply_scaled()

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        """
        鼠标左键点击即发射 clicked 信号，将原始 QPixmap 传出去。
        """
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self._orig_pm is not None:
            self.clicked.emit(self._orig_pm)
        super().mousePressEvent(e)

    def _apply_scaled(self):
        """
        内部：基于 _orig_pm 和当前控件尺寸，生成平滑的等比缩放 QPixmap 并显示。
        注意：为了避免像素数据悬空，调用 setPixmap 时使用“拷贝”或 Qt 自身的缓存机制。
        """
        if self._orig_pm is None:
            return
        scaled = self._orig_pm.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


# =============================================================================
# 图像预/后处理工具函数
# =============================================================================
def letterbox_square(img_bgr: np.ndarray, size: int, pad_value=0):
    """
    将任意 H×W 图像进行“等比缩小 + 补边”得到正方形 (size, size)：
    - 计算 scale = size / max(h, w)，并且不放大（保证 <= 1.0）；
    - 按比例缩小到 (nh, nw)，再在四周补边到 (size, size)；
    - 记录下 scale 和 pad（top, bottom, left, right），供推理后还原。
    """
    h, w = img_bgr.shape[:2]
    scale = min(size / max(h, w), 1.0)            # 不放大（避免虚化）
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
    还原 letterbox：
    - 从 (size, size) 的概率图/掩膜中，去除补边区域；
    - 以 1/scale 缩放回原始图像尺寸 (oh, ow)。
    """
    top, bottom, left, right = pad
    size = mask_sq.shape[0]
    crop = mask_sq[top:size - bottom, left:size - right]  # 去补边
    oh, ow = orig_hw
    if crop.shape[:2] == (oh, ow):
        return crop
    return cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)


def to_qpixmap_from_bgr(img_bgr: np.ndarray) -> QtGui.QPixmap:
    """
    OpenCV 的 BGR(ndarray) -> Qt 的 QPixmap：
    - OpenCV 以 BGR 存储；Qt 需要 RGB 排列；
    - QImage 持有外部数据指针，为避免生命周期问题，这里用 .copy() 拷贝一份。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    qimg = QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def make_overlay(img_bgr: np.ndarray, mask01: np.ndarray, color=(0, 0, 255), alpha=0.35):
    """
    将 mask>0.5 的区域涂上半透明颜色（默认红色），与原图按 alpha 叠加。
    - 这样可直观观察分割区域与原图对齐情况；
    - alpha 越大覆盖越明显，建议 0.3~0.5。
    """
    base = img_bgr.copy()
    overlay = np.zeros_like(base, dtype=np.uint8)
    overlay[mask01 > 0.5] = color
    return cv2.addWeighted(base, 1.0, overlay, alpha, 0)


# =============================================================================
# 多目标几何后处理：连通域 -> 轮廓 -> 旋转矩形(minAreaRect) -> 提取对角线
# =============================================================================
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
    概率图（0..1） -> 二值（0/255） -> 形态学清理 -> 连通域筛选 -> 对每个连通域拟合旋转矩形：
    - 使用 cv2.connectedComponentsWithStats 以像素面积筛除小噪点；
    - 对保留的每个连通域提取外轮廓，并用 cv2.minAreaRect 计算旋转矩形；
    - 通过 cv2.boxPoints 得到 4 角点（顺序不定！），从 4 点所有“成对距离”中找出“两条最长且不共点”的线段，
      这两条线段就是矩形的对角线（能避免把边当成对角线的误判）。
    返回：
      clean_bin: 清理后的二值图 (H,W) uint8，值为 0/255；
      detections: list[dict]，按面积从大到小排序，每项包括：
          {
            "label": int 连通域标签,
            "area": int 连通域面积（像素）,
            "box": np.ndarray(4,2) 四角坐标（int32），注意非 TL/TR/BR/BL 顺序,
            "center": (cx, cy) 旋转矩形中心（float）,
            "d1": float, "d2": float, "d_mean": float   # 两条对角线长度及其均值（像素）
          }
    """
    h, w = prob01.shape[:2]
    min_area = max(200, int(min_area_frac * h * w))  # 像素最小面积阈值，至少 200，避免超小噪点

    # 1) 概率二值化
    mask = (prob01 >= bin_thresh).astype(np.uint8) * 255

    # 2) 形态学开/闭运算：先开（去毛刺/细小噪点），再闭（补小洞/连短缺口）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)

    # 3) 连通域分析（8 邻接），并按面积过滤
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

    # 4) 对每个保留连通域，提取最外层轮廓并拟合旋转矩形
    for i, area in keep_labels:
        # 单独构造该连通域的掩膜图，用于 findContours
        mask_i = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)  # 若存在多个外轮廓取面积最大者

        rect = cv2.minAreaRect(cnt)                   # ((cx,cy), (w,h), angle)
        box  = cv2.boxPoints(rect).astype(np.int32)   # 4 个角点（顺序非 TL/TR/BR/BL）

        # —— 从 4 点的所有配对距离中，取“最长且不共点”的两条线段作为对角线 ——
        pairs = []
        for a in range(4):
            for b in range(a + 1, 4):
                d = float(np.linalg.norm(box[a] - box[b]))
                pairs.append((d, a, b))
        pairs.sort(reverse=True, key=lambda x: x[0])  # 按距离降序
        _, i1, j1 = pairs[0]                          # 最长的一条
        # 第二条：必须与第一条不共享端点（剩余两点自然是另一条对角线）
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

    # 5) 为便于阅读，默认按面积从大到小排序（你也可以换成按 d_mean 排序）
    detections.sort(key=lambda x: x["area"], reverse=True)
    return clean, detections


def draw_detections_on_three(
    img_bgr: np.ndarray,
    clean_bin: np.ndarray,
    overlay_bgr: np.ndarray,
    detections: list,
):
    """
    将所有目标的几何信息（旋转矩形 + 两条对角线 + 文本标签）**同时绘制**到三张图上：
      - vis_o：原图（BGR）
      - vis_b：二值图转 BGR（便于彩色叠加绘制）
      - vis_v：半透明覆盖图

    绘制规范：
      * 外框使用颜色列表循环（便于区分多个目标）；
      * 对角线统一用红色；
      * 文本“#编号 mean=..px”绘制在目标中心点右上方。

    返回：
      vis_o, vis_b, vis_v  —— 三张 BGR 图（ndarray）。
    """
    # 一组可读性较好的调色板（循环使用）
    palette = [
        (0, 255, 0),     # 绿
        (255, 0, 0),     # 蓝（注意 OpenCV 是 BGR，这里(255,0,0)是蓝）
        (0, 255, 255),   # 黄
        (255, 0, 255),   # 品红
        (0, 165, 255),   # 橙
        (255, 255, 0),   # 青
        (147, 20, 255),  # 紫
        (50, 205, 50),   # 黄绿
    ]
    diag_color = (0, 0, 255)  # 对角线：红色（BGR）

    vis_o = img_bgr.copy()
    vis_b = cv2.cvtColor(clean_bin, cv2.COLOR_GRAY2BGR)
    vis_v = overlay_bgr.copy()

    for idx, det in enumerate(detections, start=1):
        box = det["box"].astype(np.int32)
        color_box = palette[(idx - 1) % len(palette)]

        # 再次基于“最长且不共点”规则获取两条对角线，避免依赖顶点顺序
        pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                d = float(np.linalg.norm(box[i] - box[j]))
                pairs.append((d, i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        _, i1, j1 = pairs[0]
        rest = [k for k in range(4) if k not in (i1, j1)]
        i2, j2 = rest[0], rest[1]

        # 在三张画布上统一绘制：外框 + 两条对角线 + 文本
        for canvas in (vis_o, vis_b, vis_v):
            cv2.polylines(canvas, [box.reshape(-1, 1, 2)], True, color_box, 2)
            cv2.line(canvas, tuple(box[i1]), tuple(box[j1]), diag_color, 2)
            cv2.line(canvas, tuple(box[i2]), tuple(box[j2]), diag_color, 2)

            cx, cy = int(det["center"][0]), int(det["center"][1])
            label_txt = f"#{idx} mean={det['d_mean']:.1f}px"
            cv2.putText(canvas, label_txt, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_box, 2, cv2.LINE_AA)

    return vis_o, vis_b, vis_v


# =============================================================================
# 模型封装（ONNX / PyTorch 自适应），统一得到与原图同尺寸的概率图
# =============================================================================
class Segmenter:
    """
    UNet 推理封装：
    - 自动选择 ONNX 或 PyTorch（SMP.Unet[resnet34]）；
    - 输入前处理：letterbox 到 (S,S)、BGR->RGB、归一化、HWC->CHW、加 batch；
    - 输出后处理：unletterbox 还原回原图尺寸；
    - 返回：与原图等尺寸的概率图（float32，[0..1]）。
    """
    def __init__(self, model_dir: Path, img_size: int):
        self.model_dir = Path(model_dir)
        self.img_size = img_size
        self.kind = None     # "onnx" / "pth" / None
        self._ok = False

        # 约定的模型文件名（也可扩展自动搜索）
        self.onnx_path = self.model_dir / "unet.onnx"
        self.pth_path  = self.model_dir / "last.pth"
        if not self.pth_path.exists():
            self.pth_path = self.model_dir / "best.pth"

        # 1) 优先 ONNX（若有）
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

        # 2) 其次 PyTorch（若有）
        if not self._ok and self.pth_path.exists():
            try:
                import torch
                import segmentation_models_pytorch as smp
                self.torch = torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                # 注意：以下结构与训练时保持一致
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
        """
        便于 UI 显示当前使用的模型信息。
        """
        if not self._ok:
            return "未加载模型"
        return f"ONNX: {self.onnx_path.name}" if self.kind == "onnx" else f"PTH: {self.pth_path.name} (resnet34)"

    def preprocess(self, img_bgr: np.ndarray):
        """
        预处理流水线：
        1) letterbox 到正方形 (S,S)，避免拉伸形变；
        2) BGR->RGB，并按 ImageNet 统计量做标准化（(x-mean)/std）；
        3) HWC -> CHW，并扩维成 [1,3,S,S] float32 tensor/numpy；
        4) 返回输入张量与还原所需的元信息（scale、pad、原始尺寸）。
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
        前向推理（统一接口）：
        - 对输入图像做 preprocess；
        - 分别走 ONNX 或 PyTorch 分支拿到 (S,S) 概率图；
        - 用 unletterbox 还原回原图尺寸；
        - 返回 float32，范围 [0..1]。
        """
        if not self._ok:
            raise RuntimeError("模型未加载")
        inp, meta = self.preprocess(img_bgr)
        scale, pad, orig_hw = meta

        if self.kind == "onnx":
            out = self.sess.run(None, {self.input_name: inp})[0]  # [1,1,S,S] logits
            prob_sq = 1.0 / (1.0 + np.exp(-out))                 # Sigmoid(logits)
            prob_sq = prob_sq[0, 0]                              # (S,S)
        else:
            torch = self.torch
            x = torch.from_numpy(inp).to(self.device, non_blocking=True)
            with torch.inference_mode():
                logits = self.model(x)                           # [1,1,S,S]
                prob_sq = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()

        # 还原至原图尺寸（去补边 + 等比缩放回去）
        prob = unletterbox(prob_sq.astype(np.float32), scale, pad, orig_hw)
        prob = np.clip(prob, 0.0, 1.0)
        return prob


# =============================================================================
# 主窗口：2×3 网格可视化 + 日志 + 点击弹大图
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    """
    结构：
      顶部：按钮（打开图片）、模型信息文本；
      中部：6 个 ScaledLabel（两行三列），每个都能点击弹大图；
      底部：日志（QPlainTextEdit，打印耗时和每个目标的几何统计）。
    """
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, img_size=DEFAULT_IMG_SIZE):
        super().__init__()
        self.setWindowTitle("维氏压痕识别 — 多目标 | UNet + OpenCV(minAreaRect)")
        self.resize(1680, 980)

        # —— 模型加载（按优先级选择 ONNX 或 PTH） ——
        self.segmenter = Segmenter(model_dir, img_size)

        # 顶部工具区：打开图片按钮 + 模型说明
        self.open_btn = QtWidgets.QPushButton("打开图片…")
        self.open_btn.clicked.connect(self.on_open)
        self.model_label = QtWidgets.QLabel(self.segmenter.info())
        self.model_label.setStyleSheet("color:#777;")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.open_btn)
        top.addSpacing(12)
        top.addWidget(self.model_label)
        top.addStretch(1)

        # —— 6 个视图容器（等比缩略 + 可点击） ——
        self.view_orig   = ScaledLabel()  # 原图
        self.view_mask   = ScaledLabel()  # 掩膜(灰度)
        self.view_ovl    = ScaledLabel()  # 半透明覆盖
        self.view_geom_o = ScaledLabel()  # 几何结果@原图
        self.view_geom_b = ScaledLabel()  # 几何结果@二值
        self.view_geom_v = ScaledLabel()  # 几何结果@覆盖

        # 通用外观：最小尺寸、描边
        for v in (self.view_orig, self.view_mask, self.view_ovl,
                  self.view_geom_o, self.view_geom_b, self.view_geom_v):
            v.setMinimumSize(220, 220)
            v.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        # —— 将“点击缩略图 -> 弹预览”的动作接上 ——
        # 使用 lambda 绑定不同标题，保持可读性
        self.view_orig.clicked.connect(lambda pm: self._open_preview(pm, "原图"))
        self.view_mask.clicked.connect(lambda pm: self._open_preview(pm, "掩膜(灰度)"))
        self.view_ovl.clicked.connect(lambda pm: self._open_preview(pm, "半透明覆盖"))
        self.view_geom_o.clicked.connect(lambda pm: self._open_preview(pm, "几何结果@原图"))
        self.view_geom_b.clicked.connect(lambda pm: self._open_preview(pm, "几何结果@二值"))
        self.view_geom_v.clicked.connect(lambda pm: self._open_preview(pm, "几何结果@覆盖"))

        # —— 网格布局（2 行 × 3 列） ——
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
            grid.setColumnStretch(c, 1)  # 每列等比拉伸
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # — 底部日志区 —
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMinimumHeight(160)
        self.log.setStyleSheet("font-family: Consolas, Menlo, monospace; font-size:12px;")

        # 中央大布局（垂直堆叠：顶部工具条 + 网格 + 日志）
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addLayout(top)
        lay.addLayout(grid, 1)   # 网格区域占优先空间
        lay.addWidget(self.log)
        self.setCentralWidget(central)

        # 若未成功加载模型，给出一次友好提示，避免用户误会
        if not self.segmenter.ok():
            self.append_log(f"[WARN] 未找到模型：{model_dir}")
            self.append_log("请将 unet.onnx 或 best/last.pth 放入该目录。")

    # --- 工具方法：为子控件加标题包裹（GroupBox） ---
    def _titled(self, title, widget):
        """
        将任意控件用 QGroupBox 包裹，并在顶部显示标题文字。
        """
        box = QtWidgets.QGroupBox(title)
        l = QtWidgets.QVBoxLayout(box)
        l.addWidget(widget)
        return box

    # --- 工具方法：日志带时间戳 ---
    def append_log(self, text: str):
        """
        向底部日志输出一行文本，并自动附上当前时分秒。
        """
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {text}")

    # --- 点击缩略图时打开预览弹窗 ---
    def _open_preview(self, pm: QtGui.QPixmap, title: str):
        """
        响应 6 张缩略图的 clicked(QPixmap) 信号：
        - 使用非模态弹窗（.show()），可同时打开多个；
        - 若你更习惯模态（阻塞），把 .show() 改成 .exec() 即可。
        """
        dlg = ZoomImageDialog(pm, title=title, parent=self)
        dlg.show()

    # --- 打开图片并完成一次推理与可视化 ---
    @QtCore.Slot()
    def on_open(self):
        """
        流程：
          1) 文件对话框选择图片；
          2) 读取原图并显示到“原图”位；
          3) 若模型未加载，仅提示并返回；
          4) 执行模型推理（统计耗时）；
          5) 构建第一行：掩膜(灰度)、半透明覆盖；
          6) 调用 postprocess_minarearect_multi 进行几何后处理；
          7) 绘制第二行的三张“几何结果图”；
          8) 在日志中打印耗时与每个目标的几何统计值。
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return

        # 计时（总耗时的起点）
        t0 = time.perf_counter()
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.append_log(f"读取失败：{path}")
            return

        # 第一行：原图
        self.view_orig.setPixmap(to_qpixmap_from_bgr(img))

        # 若未加载模型，则跳过推理与后处理
        if not self.segmenter.ok():
            self.append_log("模型未加载，仅显示原图。")
            return

        # —— 推理（统计单独耗时） ——
        t1 = time.perf_counter()
        prob = self.segmenter.infer(img)  # (H,W) float32 [0..1]
        t2 = time.perf_counter()

        # 第一行：掩膜灰度
        mask_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        self.view_mask.setPixmap(to_qpixmap_from_bgr(mask_bgr))

        # 第一行：半透明覆盖（红色 overlay）
        overlay = make_overlay(img, prob, color=(0, 0, 255), alpha=0.35)
        self.view_ovl.setPixmap(to_qpixmap_from_bgr(overlay))

        # —— 几何后处理（多目标） ——
        clean, detections = postprocess_minarearect_multi(
            img, prob,
            bin_thresh=BIN_THRESH,
            min_area_frac=MIN_AREA_FRAC,
            morph_kernel=MORPH_KERNEL,
            open_iter=OPEN_ITER,
            close_iter=CLOSE_ITER,
        )

        # 第二行：在三张图上叠加绘制所有检测到的旋转矩形/对角线/文本
        vis_o, vis_b, vis_v = draw_detections_on_three(img, clean, overlay, detections)
        self.view_geom_o.setPixmap(to_qpixmap_from_bgr(vis_o))
        self.view_geom_b.setPixmap(to_qpixmap_from_bgr(vis_b))
        self.view_geom_v.setPixmap(to_qpixmap_from_bgr(vis_v))

        # —— 日志输出：文件信息与耗时统计 ——
        self.append_log(f"打开：{Path(path).name}  尺寸={img.shape[1]}×{img.shape[0]}")
        self.append_log(
            f"预处理+I/O: {(t1 - t0)*1e3:.1f} ms | 推理: {(t2 - t1)*1e3:.1f} ms | 总计: {(t2 - t0)*1e3:.1f} ms"
        )
        self.append_log(f"模型：{self.segmenter.info()}  img_size={self.segmenter.img_size}")

        # —— 日志输出：几何检测的统计信息（逐个目标） ——
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


# =============================================================================
# 入口（标准 Qt 程序样板）
# =============================================================================
def main():
    """
    Qt 应用启动入口：
    - 创建 QApplication；
    - 构造主窗口 MainWindow（传入默认模型目录与输入尺寸）；
    - 显示窗口并进入事件循环。
    """
    app = QtWidgets.QApplication([])
    w = MainWindow(model_dir=DEFAULT_MODEL_DIR, img_size=DEFAULT_IMG_SIZE)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
