# -*- coding: utf-8 -*-
"""
ui_infer.py  —— 维氏压痕多目标分割 + 四边形拟合（带大图预览）
================================================================================
功能概述（高可读注释版）：
- 使用 U-Net（ONNX 或 PyTorch）对维氏压痕进行语义分割，得到每像素概率图；
- 概率图经阈值/形态学/连通域 -> 对每个目标进行**鲁棒四边形拟合**（替代 minAreaRect）：
  * 凸包抑制鼓边；approxPolyDP + 二分 epsilon 收敛到 4 点；
  * 失败则从 >4 点结果做子采样候选；最后以极值点兜底；
  * 输出顺时针四边形，并统计两条对角线和其均值；
- UI：2×3 网格（原/掩/覆盖 | 结果@原/结果@二值/结果@覆盖）；底部日志显示耗时与结果；
- 交互：点击任意缩略图弹出**大图预览**（QGraphicsView，支持滚轮缩放、拖拽、双击复位）。

模型加载优先级：
1) 目录存在 unet.onnx ——> ONNX Runtime（优先 CUDAExecutionProvider，再回退 CPU）
2) 否则若存在 last.pth / best.pth ——> PyTorch + segmentation_models_pytorch.Unet(resnet34)
3) 两者都无 ——> 不做推理，仅显示原图并提示

注意：
- DEFAULT_IMG_SIZE 必须与训练时一致（如 512），否则可能输入不匹配或精度下降。
- 统一采用 letterbox（等比缩放+补边）适配模型输入；推理后用 unletterbox 还原尺寸。
"""

from pathlib import Path
import time
import math
import numpy as np
import cv2
from PySide6 import QtCore, QtGui, QtWidgets


# =============================================================================
#                        1) 可调参数（按需修改）
# =============================================================================
DEFAULT_MODEL_DIR = Path("runs/unet_r34_512")   # 模型目录：放 unet.onnx 或 best/last.pth
DEFAULT_IMG_SIZE  = 512                         # 模型输入正方形尺寸（需与训练一致）
IMAGENET_MEAN     = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # 归一化均值
IMAGENET_STD      = np.array([0.229, 0.224, 0.225], dtype=np.float32)  # 归一化方差

# 后处理默认参数（可在 on_open() 里统一调整）
BIN_THRESH       = 0.50    # 概率二值化阈值；保守可降至 0.45；粘连多可升至 0.55
MIN_AREA_FRAC    = 0.0008  # 最小连通域面积占比（相对整图）；分辨率高/噪点多可调到 0.001~0.003
MORPH_KERNEL     = 3       # 形态学核大小（奇数 3/5/7）；纹理噪点多用 5
OPEN_ITER        = 1       # 开运算迭代（去毛刺/小噪点）
CLOSE_ITER       = 1       # 闭运算迭代（连通缺口）


# =============================================================================
#                        2) 大图预览（QGraphicsView）
# =============================================================================
class ZoomImageDialog(QtWidgets.QDialog):
    """
    大图预览弹窗：
    - 使用 QGraphicsView 展示 QPixmap；
    - 滚轮缩放（以鼠标为中心）、拖拽平移、双击复位（适配窗口）。
    """
    def __init__(self, pixmap: QtGui.QPixmap, title: str = "预览", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1024, 720)
        self.setWindowModality(QtCore.Qt.WindowModality.NonModal)

        self._view = _GraphicsView(self)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._item = QtWidgets.QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._item)
        self._view.setScene(self._scene)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._view)

        QtCore.QTimer.singleShot(0, self._view.fitInViewSmooth)

    def setPixmap(self, pm: QtGui.QPixmap):
        self._item.setPixmap(pm)
        self._view.fitInViewSmooth()


class _GraphicsView(QtWidgets.QGraphicsView):
    """带滚轮缩放/拖拽/双击复位的轻定制视图。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing |
                            QtGui.QPainter.SmoothPixmapTransform |
                            QtGui.QPainter.TextAntialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._min_scale = 0.05
        self._max_scale = 50.0

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 1 / 1.25
        cur_scale = self.transform().m11()
        new_scale = cur_scale * factor
        if new_scale < self._min_scale and factor < 1:
            factor = self._min_scale / max(cur_scale, 1e-9)
        elif new_scale > self._max_scale and factor > 1:
            factor = self._max_scale / max(cur_scale, 1e-9)
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        self.resetTransform()
        self.fitInViewSmooth()

    def fitInViewSmooth(self):
        r = self.sceneRect()
        if not r.isNull():
            self.fitInView(r, QtCore.Qt.AspectRatioMode.KeepAspectRatio)


# =============================================================================
#                   3) 缩略图控件（等比缩放 + 点击信号）
# =============================================================================
class ScaledLabel(QtWidgets.QLabel):
    """
    可等比缩放的 QLabel：
    - setPixmap() 记录原始 QPixmap（未缩放），resize 时平滑缩放显示；
    - 左键单击发射 clicked(QPixmap) 信号，用于弹出预览。
    """
    clicked = QtCore.Signal(QtGui.QPixmap)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orig_pm: QtGui.QPixmap | None = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def setPixmap(self, pm: QtGui.QPixmap) -> None:  # type: ignore[override]
        self._orig_pm = pm
        self._apply_scaled()

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(e)
        if self._orig_pm is not None:
            self._apply_scaled()

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if e.button() == QtCore.Qt.MouseButton.LeftButton and self._orig_pm is not None:
            self.clicked.emit(self._orig_pm)
        super().mousePressEvent(e)

    def _apply_scaled(self):
        if self._orig_pm is None:
            return
        scaled = self._orig_pm.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


# =============================================================================
#                     4) 基础图像工具（letterbox / overlay）
# =============================================================================
def letterbox_square(img_bgr: np.ndarray, size: int, pad_value=0):
    """
    任意 H×W 图像 -> 等比缩小 + 补边成 (size,size)：
    - 不放大（避免虚化），记录 scale 和 pad（top/bottom/left/right）。
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
    还原 letterbox：去补边并按 1/scale 缩回原始尺寸。
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
    OpenCV BGR(ndarray) -> Qt QPixmap（RGB 顺序 + 拷贝内存确保生命周期安全）。
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    qimg = QtGui.QImage(img_rgb.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def make_overlay(img_bgr: np.ndarray, mask01: np.ndarray, color=(0, 0, 255), alpha=0.35):
    """
    在 mask>0.5 区域叠加半透明颜色层（默认红色），直观查看分割与原图对齐情况。
    """
    base = img_bgr.copy()
    overlay = np.zeros_like(base, dtype=np.uint8)
    overlay[mask01 > 0.5] = color
    return cv2.addWeighted(base, 1.0, overlay, alpha, 0)


# =============================================================================
#           5) 四边形拟合相关的小工具（排序/凸性/面积/评分）
# =============================================================================
def _order_quad_cw(pts: np.ndarray) -> np.ndarray:
    """
    将 4 点按**顺时针**排序，并固定起点（y 最小；并列则 x 最小）以稳定输出。
    """
    p = pts.astype(np.float32).reshape(-1, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    idx = np.argsort(ang)          # 逆时针
    p = p[idx[::-1]]               # 顺时针
    k = np.lexsort((p[:, 0], p[:, 1]))[0]
    return np.roll(p, -k, axis=0)


def _is_convex_quad(p: np.ndarray) -> bool:
    """
    简单凸性：顺时针四边的叉积符号应一致。
    """
    p = p.reshape(4, 2)
    sgn = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        v1 = b - a
        v2 = c - b
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sgn.append(cross)
    return (all(x >= 0 for x in sgn) or all(x <= 0 for x in sgn))


def _poly_area(p: np.ndarray) -> float:
    """多边形面积（返回正值）。"""
    x, y = p[:, 0], p[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def _quad_quality(p: np.ndarray) -> float:
    """
    四边形“形状质量”评分（0~1，越大越好），轻度偏好：
    - 角度不过尖（>15°）不过钝（<165°）；
    - 边长均匀（max/min 接近 1）；
    - 周长越大略好（抑制退化）。
    """
    p = p.reshape(4, 2)
    dists = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
    peri = sum(dists) + 1e-6
    # 角度
    penalties = []
    for i in range(4):
        a, b, c = p[(i - 1) % 4], p[i], p[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        ang = math.degrees(math.acos(np.clip(cos, -1, 1)))
        penalties.append(0.0 if 15.0 <= ang <= 165.0 else 1.0)
    ang_pen = np.mean(penalties)
    ed_ratio = (max(dists) + 1e-6) / (min(dists) + 1e-6)
    ed_pen = min(1.0, abs(ed_ratio - 1.0))
    return (1.0 - 0.5 * ang_pen) * (1.0 - 0.5 * ed_pen) * (peri / (peri + 1000.0))


# =============================================================================
#             6) 核心：从轮廓鲁棒地拟合 **四边形**（替代 minAreaRect）
# =============================================================================
def robust_quadrilateral_from_contour(cnt: np.ndarray,
                                      want_convex: bool = True,
                                      max_iter: int = 25) -> np.ndarray | None:
    """
    将一个轮廓（N,1,2 或 N,2）鲁棒拟合为**四边形**：
    0) 抑制鼓边/毛刺：先对点集取**凸包**；
    1) 对“原轮廓”和“凸包轮廓”分别进行 approxPolyDP 的 **epsilon 二分搜索**，
       尝试收敛到**恰好 4 点**；
    2) 若直接得到 4 点，检查**面积/凸性**，合格返回；
    3) 若只得到 >4 点，则做**子采样**构造四边形候选；
    4) 若仍失败，以**极值点**（x_min/x_max/y_min/y_max）兜底；
    5) 在所有候选中，按“形状质量 + 面积”选最优。
    """
    pts = cnt.reshape(-1, 2).astype(np.float32)
    if pts.shape[0] < 4:
        return None

    # —— 凸包：将外凸“鼓边”抹平，减少对近似的干扰 ——
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float32)

    def _try_poly_dp(poly: np.ndarray) -> np.ndarray | None:
        """对 poly 做 approxPolyDP 的 epsilon 二分搜索（目标恰好 4 点）。"""
        peri = cv2.arcLength(poly.reshape(-1, 1, 2), True)
        lo, hi = 0.001 * peri, 0.08 * peri  # 经验范围：过小点多、过大点少
        best4 = None
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            appr = cv2.approxPolyDP(poly.reshape(-1, 1, 2), epsilon=mid, closed=True)
            n = len(appr)
            if n == 4:
                cand = _order_quad_cw(appr.reshape(-1, 2).astype(np.float32))
                if _poly_area(cand) > 10 and (not want_convex or _is_convex_quad(cand)):
                    best4 = cand
                    # 不强求更小 epsilon（更贴合），当前已可用；也可 hi=mid 继续逼近
                    break
                else:
                    lo = mid  # 形状不佳，尝试更大 epsilon（更内收）
            elif n > 4:
                lo = mid  # 点太多 -> 增大 epsilon
            else:
                hi = mid  # 点太少 -> 减小 epsilon
            if abs(hi - lo) < 1e-6:
                break
        return best4

    # ① 在“原轮廓 + 凸包”两条路径尝试
    candidates = []
    for poly in (pts, hull):
        got = _try_poly_dp(poly)
        if got is not None:
            candidates.append(got)

    # ② 没拿到 4 点？从 >4 点近似结果中，做**间隔子采样**构造候选
    if not candidates:
        for poly in (pts, hull):
            peri = cv2.arcLength(poly.reshape(-1, 1, 2), True)
            eps = 0.01 * peri
            appr = cv2.approxPolyDP(poly.reshape(-1, 1, 2), eps, True).reshape(-1, 2).astype(np.float32)
            k = len(appr)
            if k > 4:
                for s in range(0, min(12, k)):  # 最多尝试 12 组起点，线性复杂度
                    idx = np.arange(s, s + 4) % k
                    cand = _order_quad_cw(appr[idx])
                    if _poly_area(cand) > 10 and (not want_convex or _is_convex_quad(cand)):
                        candidates.append(cand)

    # ③ 仍无候选？以**极值点**兜底（从凸包取 min/max 的四个极值）
    if not candidates:
        xs, ys = hull[:, 0], hull[:, 1]
        i_minx, i_maxx = int(np.argmin(xs)), int(np.argmax(xs))
        i_miny, i_maxy = int(np.argmin(ys)), int(np.argmax(ys))
        raw = np.array([hull[i_miny], hull[i_maxx], hull[i_maxy], hull[i_minx]], np.float32)
        cand = _order_quad_cw(raw)
        if _poly_area(cand) > 10:
            candidates.append(cand)

    if not candidates:
        return None

    # ④ 候选中选最优（先形状质量，再面积）
    candidates.sort(key=lambda q: (_quad_quality(q), _poly_area(q)), reverse=True)
    return candidates[0]


# =============================================================================
#     7) 多目标后处理（使用“四边形拟合”替代 minAreaRect；接口保持不变）
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
    ——【四边形拟合版】——
    概率图 -> 二值 -> 形态学清理 -> 连通域筛选 -> 对每个连通域拟合“鲁棒四边形”
    （替代原先的 minAreaRect；函数名保持不变，便于无缝替换其它调用）。

    返回：
      clean_bin: 清理后的二值图 (uint8, 0/255)
      detections: list[dict]（按面积降序）：
          {
            "label": int,                    # 连通域标签
            "area": int,                    # 连通域面积（像素）
            "box": np.ndarray(4,2),         # 四边形顶点，顺时针（int32）
            "center": (cx, cy),             # 中心（顶点均值）
            "d1": float, "d2": float,       # 两条对角线长度（像素）
            "d_mean": float                 # 对角线均值
          }
    """
    h, w = prob01.shape[:2]
    min_area = max(200, int(min_area_frac * h * w))

    # 1) 概率二值化
    mask = (prob01 >= bin_thresh).astype(np.uint8) * 255

    # 2) 形态学清理（先开后闭）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_iter)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_iter)

    # 3) 连通域筛选
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    keep_labels = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == i] = 255
            keep_labels.append((i, area))

    detections = []
    if not keep_labels:
        return clean, detections

    # 4) 对每个连通域：轮廓 -> 鲁棒四边形 -> 统计
    for i, area in keep_labels:
        mask_i = (labels == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)

        quad = robust_quadrilateral_from_contour(cnt, want_convex=True)
        if quad is None:
            continue
        quad = _order_quad_cw(quad).astype(np.int32)

        # 中心点（顶点均值）
        cx, cy = float(np.mean(quad[:, 0])), float(np.mean(quad[:, 1]))

        # 对角线：在 4 点中选“最长且不共点”的两条（防止把边当对角线）
        pairs = []
        for a in range(4):
            for b in range(a + 1, 4):
                pairs.append((float(np.linalg.norm(quad[a] - quad[b])), a, b))
        pairs.sort(reverse=True, key=lambda x: x[0])
        _, i1, j1 = pairs[0]
        rest = [k for k in range(4) if k not in (i1, j1)]
        i2, j2 = rest[0], rest[1]
        d1 = float(np.linalg.norm(quad[i1] - quad[j1]))
        d2 = float(np.linalg.norm(quad[i2] - quad[j2]))

        detections.append({
            "label": i,
            "area": area,
            "box": quad,
            "center": (cx, cy),
            "d1": d1, "d2": d2, "d_mean": 0.5 * (d1 + d2),
        })

    detections.sort(key=lambda x: x["area"], reverse=True)
    return clean, detections


# =============================================================================
#                         8) 绘制：三张图同步画结果
# =============================================================================
def draw_detections_on_three(
    img_bgr: np.ndarray,
    clean_bin: np.ndarray,
    overlay_bgr: np.ndarray,
    detections: list,
):
    """
    将所有检测结果（四边形 + 两条对角线 + 文本标签）同步绘制到：
      - vis_o：原图
      - vis_b：二值图（转 BGR）
      - vis_v：覆盖图
    """
    palette = [
        (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
        (0, 165, 255), (255, 255, 0), (147, 20, 255), (50, 205, 50),
    ]
    diag_color = (0, 0, 255)  # 对角线红色（BGR）

    vis_o = img_bgr.copy()
    vis_b = cv2.cvtColor(clean_bin, cv2.COLOR_GRAY2BGR)
    vis_v = overlay_bgr.copy()

    for idx, det in enumerate(detections, start=1):
        quad = det["box"].astype(np.int32)
        color_box = palette[(idx - 1) % len(palette)]

        # 从四点组合里找两条“最长且不共点”的对角线
        pairs = []
        for i in range(4):
            for j in range(i + 1, 4):
                pairs.append((float(np.linalg.norm(quad[i] - quad[j])), i, j))
        pairs.sort(reverse=True, key=lambda x: x[0])
        _, i1, j1 = pairs[0]
        rest = [k for k in range(4) if k not in (i1, j1)]
        i2, j2 = rest[0], rest[1]

        for canvas in (vis_o, vis_b, vis_v):
            cv2.polylines(canvas, [quad.reshape(-1, 1, 2)], True, color_box, 2)
            cv2.line(canvas, tuple(quad[i1]), tuple(quad[j1]), diag_color, 2)
            cv2.line(canvas, tuple(quad[i2]), tuple(quad[j2]), diag_color, 2)
            cx, cy = int(det["center"][0]), int(det["center"][1])
            cv2.putText(canvas, f"#{idx} mean={det['d_mean']:.1f}px",
                        (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        color_box, 2, cv2.LINE_AA)

    return vis_o, vis_b, vis_v


# =============================================================================
#                           9) 模型封装（ONNX/PTH）
# =============================================================================
class Segmenter:
    """
    UNet 推理封装：
    - 自动选择 ONNX 或 PyTorch（SMP.Unet(resnet34)）；
    - 统一前处理（letterbox、归一化、CHW、batch）与后处理（unletterbox）；
    - 输出与原图同尺寸的概率图（float32，范围 [0..1]）。
    """
    def __init__(self, model_dir: Path, img_size: int):
        self.model_dir = Path(model_dir)
        self.img_size = img_size
        self.kind = None
        self._ok = False

        self.onnx_path = self.model_dir / "unet.onnx"
        self.pth_path  = self.model_dir / "last.pth"
        if not self.pth_path.exists():
            self.pth_path = self.model_dir / "best.pth"

        # 优先 ONNX
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

        # 回退 PyTorch
        if not self._ok and self.pth_path.exists():
            try:
                import torch
                import segmentation_models_pytorch as smp
                self.torch = torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        前处理：
        - letterbox 到 (S,S)
        - BGR->RGB、归一化到 [0,1]，再 (x-mean)/std
        - HWC -> CHW，扩展 batch 维 -> [1,3,S,S]
        """
        h0, w0 = img_bgr.shape[:2]
        sq, scale, pad = letterbox_square(img_bgr, self.img_size, pad_value=0)
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(rgb, (2, 0, 1))
        inp = np.expand_dims(chw, 0).astype(np.float32)  # [1,3,S,S]
        return inp, (scale, pad, (h0, w0))

    def infer(self, img_bgr: np.ndarray) -> np.ndarray:
        """统一推理接口：返回与原图同尺寸的概率图（float32，[0..1]）。"""
        if not self._ok:
            raise RuntimeError("模型未加载")
        inp, meta = self.preprocess(img_bgr)
        scale, pad, orig_hw = meta

        if self.kind == "onnx":
            out = self.sess.run(None, {self.input_name: inp})[0]  # [1,1,S,S] logits
            prob_sq = 1.0 / (1.0 + np.exp(-out))
            prob_sq = prob_sq[0, 0]                               # (S,S)
        else:
            torch = self.torch
            x = torch.from_numpy(inp).to(self.device, non_blocking=True)
            with torch.inference_mode():
                logits = self.model(x)                            # [1,1,S,S]
                prob_sq = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()

        prob = unletterbox(prob_sq.astype(np.float32), scale, pad, orig_hw)
        return np.clip(prob, 0.0, 1.0)


# =============================================================================
#                               10) 主窗口
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    """
    布局：
      顶部：打开图片按钮 + 模型信息；
      中部：6 个可点击缩略图（2×3）；
      底部：日志（QPlainTextEdit）。
    """
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, img_size=DEFAULT_IMG_SIZE):
        super().__init__()
        self.setWindowTitle("维氏压痕识别 — 多目标 | UNet + 四边形拟合")
        self.resize(1680, 980)

        # 模型
        self.segmenter = Segmenter(model_dir, img_size)

        # 顶部工具条
        self.open_btn = QtWidgets.QPushButton("打开图片…")
        self.open_btn.clicked.connect(self.on_open)
        self.model_label = QtWidgets.QLabel(self.segmenter.info())
        self.model_label.setStyleSheet("color:#777;")

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.open_btn)
        top.addSpacing(12)
        top.addWidget(self.model_label)
        top.addStretch(1)

        # 6 个缩略图（带边框）
        self.view_orig   = ScaledLabel()
        self.view_mask   = ScaledLabel()
        self.view_ovl    = ScaledLabel()
        self.view_geom_o = ScaledLabel()
        self.view_geom_b = ScaledLabel()
        self.view_geom_v = ScaledLabel()

        for v in (self.view_orig, self.view_mask, self.view_ovl,
                  self.view_geom_o, self.view_geom_b, self.view_geom_v):
            v.setMinimumSize(220, 220)
            v.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        # 点击弹预览
        self.view_orig.clicked.connect(lambda pm: self._open_preview(pm, "原图"))
        self.view_mask.clicked.connect(lambda pm: self._open_preview(pm, "掩膜(灰度)"))
        self.view_ovl.clicked.connect(lambda pm: self._open_preview(pm, "半透明覆盖"))
        self.view_geom_o.clicked.connect(lambda pm: self._open_preview(pm, "几何结果@原图"))
        self.view_geom_b.clicked.connect(lambda pm: self._open_preview(pm, "几何结果@二值"))
        self.view_geom_v.clicked.connect(lambda pm: self._open_preview(pm, "几何结果@覆盖"))

        # 网格布局
        grid = QtWidgets.QGridLayout()
        grid.addWidget(self._titled("原图", self.view_orig),         0, 0)
        grid.addWidget(self._titled("掩膜(灰度)", self.view_mask),   0, 1)
        grid.addWidget(self._titled("半透明覆盖", self.view_ovl),    0, 2)
        grid.addWidget(self._titled("几何结果@原图", self.view_geom_o), 1, 0)
        grid.addWidget(self._titled("几何结果@二值", self.view_geom_b), 1, 1)
        grid.addWidget(self._titled("几何结果@覆盖", self.view_geom_v), 1, 2)
        for c in range(3):
            grid.setColumnStretch(c, 1)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        # 日志
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
        box = QtWidgets.QGroupBox(title)
        l = QtWidgets.QVBoxLayout(box)
        l.addWidget(widget)
        return box

    def append_log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {text}")

    def _open_preview(self, pm: QtGui.QPixmap, title: str):
        dlg = ZoomImageDialog(pm, title=title, parent=self)
        dlg.show()  # 若想阻塞：改为 dlg.exec()

    @QtCore.Slot()
    def on_open(self):
        """
        选择图片 -> 显示原图 -> 推理 -> 第一行（掩膜/覆盖）-> 四边形后处理 ->
        第二行（结果三图）-> 日志输出。
        """
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

        # 推理
        t1 = time.perf_counter()
        prob = self.segmenter.infer(img)  # float32 (H,W) [0..1]
        t2 = time.perf_counter()

        # 第一行：掩膜灰度 & 覆盖
        mask_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        self.view_mask.setPixmap(to_qpixmap_from_bgr(mask_bgr))

        overlay = make_overlay(img, prob, color=(0, 0, 255), alpha=0.35)
        self.view_ovl.setPixmap(to_qpixmap_from_bgr(overlay))

        # 四边形多目标后处理
        clean, detections = postprocess_minarearect_multi(
            img, prob,
            bin_thresh=BIN_THRESH,
            min_area_frac=MIN_AREA_FRAC,
            morph_kernel=MORPH_KERNEL,
            open_iter=OPEN_ITER,
            close_iter=CLOSE_ITER,
        )

        # 第二行：绘制三张结果图
        vis_o, vis_b, vis_v = draw_detections_on_three(img, clean, overlay, detections)
        self.view_geom_o.setPixmap(to_qpixmap_from_bgr(vis_o))
        self.view_geom_b.setPixmap(to_qpixmap_from_bgr(vis_b))
        self.view_geom_v.setPixmap(to_qpixmap_from_bgr(vis_v))

        # 日志
        self.append_log(f"打开：{Path(path).name}  尺寸={img.shape[1]}×{img.shape[0]}")
        self.append_log(
            f"预处理+I/O: {(t1 - t0)*1e3:.1f} ms | 推理: {(t2 - t1)*1e3:.1f} ms | 总计: {(t2 - t0)*1e3:.1f} ms"
        )
        self.append_log(f"模型：{self.segmenter.info()}  img_size={self.segmenter.img_size}")

        if not detections:
            self.append_log("后处理：未检测到压痕。可调 bin_thresh / min_area_frac / morph_kernel。")
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
#                                 11) 入口
# =============================================================================
def main():
    app = QtWidgets.QApplication([])
    w = MainWindow(model_dir=DEFAULT_MODEL_DIR, img_size=DEFAULT_IMG_SIZE)
    w.show()
    app.exec()


if __name__ == "__main__":
    main()
