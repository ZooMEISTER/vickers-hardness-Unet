# -*- coding: utf-8 -*-
"""
ui_infer.py — Simple WPF-like desktop UI with three panes:
[Original | Mask | Overlay] + a log panel with timing.

Priority:
1) If unet.onnx exists -> use ONNX Runtime
2) Else if best.pth/last.pth exists -> use PyTorch (smp.Unet)
3) Else only show images & log error

- Top button "Open Image…" pops a file dialog.
- Left shows the original image.
- Middle shows predicted mask (grayscale).
- Right shows red-semitransparent overlay.

You can change DEFAULT_MODEL_DIR / DEFAULT_IMG_SIZE if needed.
"""

import os
import time
from pathlib import Path

import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets

# ----------------------------
# Config
# ----------------------------
DEFAULT_MODEL_DIR = Path("runs/unet_r34_384")   # where best.pth / unet.onnx saved
DEFAULT_IMG_SIZE  = 512                  # must match training cfg
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------
# Utilities
# ----------------------------

# Auto-scaling QLabel
class ScaledLabel(QtWidgets.QLabel):
    """Keep-aspect-ratio, smooth scaling on resize."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._orig_pm: QtGui.QPixmap | None = None
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # 让控件可扩展，随布局拉伸
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
        if self.width() <= 2 or self.height() <= 2:
            super().setPixmap(self._orig_pm)
            return
        scaled = self._orig_pm.scaled(
            self.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


def letterbox_square(img_bgr: np.ndarray, size: int, pad_value=0):
    """
    Resize longest side to <= size, keep aspect ratio, then pad to (size,size).
    Returns: img_pad (H,W,3 BGR), scale (float), pad (top,bottom,left,right)
    """
    h, w = img_bgr.shape[:2]
    scale = min(size / max(h, w), 1.0)  # downscale only (match training)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_rs = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left

    img_pad = cv2.copyMakeBorder(img_rs, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value))
    return img_pad, scale, (top, bottom, left, right)


def unletterbox(mask_sq: np.ndarray, scale: float, pad: tuple, orig_hw: tuple):
    """
    Inverse of letterbox_square for mask: remove pad, resize back to original HxW.
    mask_sq: (size,size) float32 [0..1]
    """
    top, bottom, left, right = pad
    size = mask_sq.shape[0]
    crop = mask_sq[top:size - bottom, left:size - right]
    oh, ow = orig_hw
    if crop.shape[0] == oh and crop.shape[1] == ow:
        return crop
    return cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)


def to_qpixmap_from_bgr(img_bgr: np.ndarray) -> QtGui.QPixmap:
    """Convert BGR (H,W,3) to QPixmap"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    qimg = QtGui.QImage(img_rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg.copy())


def make_overlay(img_bgr: np.ndarray, mask01: np.ndarray, color=(0, 0, 255), alpha=0.35):
    """
    Overlay red semi-transparent on mask>0.5 area.
    img_bgr: (H,W,3) uint8
    mask01 : (H,W) float32 [0..1]
    """
    base = img_bgr.copy()
    overlay = np.zeros_like(base, dtype=np.uint8)
    m = (mask01 > 0.5)
    overlay[m] = color
    out = cv2.addWeighted(base, 1.0, overlay, alpha, 0)
    return out


# ----------------------------
# Model Wrapper (ONNX or PyTorch)
# ----------------------------
class Segmenter:
    def __init__(self, model_dir: Path, img_size: int):
        self.model_dir = Path(model_dir)
        self.img_size = img_size
        self.kind = None

        self.onnx_path = self.model_dir / "unet.onnx"
        self.pth_path = self.model_dir / "last.pth"
        if not self.pth_path.exists():
            self.pth_path = self.model_dir / "best.pth"
        self.encoder   = "resnet34"  # must match training for .pth
        self._ok = False

        # Try ONNX first
        if self.onnx_path.exists():
            try:
                import onnxruntime as ort
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.sess = ort.InferenceSession(str(self.onnx_path), providers=providers)
                self.input_name = self.sess.get_inputs()[0].name
                self.output_name = self.sess.get_outputs()[0].name
                self.kind = "onnx"
                self._ok = True
            except Exception as e:
                self.sess = None
                print("[WARN] ONNX load failed:", e)

        # Fallback to PyTorch .pth
        if not self._ok and self.pth_path.exists():
            try:
                import torch
                import segmentation_models_pytorch as smp
                self.torch = torch
                self.smp = smp
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model = smp.Unet(
                    encoder_name=self.encoder, encoder_weights=None,
                    in_channels=3, classes=1, activation=None
                ).to(self.device)
                sd = torch.load(self.pth_path, map_location=self.device)
                self.model.load_state_dict(sd, strict=True)
                self.model.eval()
                self.kind = "pth"
                self._ok = True
            except Exception as e:
                self.model = None
                print("[WARN] PTH load failed:", e)

    def ok(self) -> bool:
        return self._ok

    def info(self) -> str:
        if not self._ok:
            return "No model loaded."
        if self.kind == "onnx":
            return f"ONNX Runtime: {self.onnx_path.name}"
        else:
            return f"PyTorch PTH: {self.pth_path.name} (encoder={self.encoder})"

    def preprocess(self, img_bgr: np.ndarray):
        """
        Return: input_tensor [1,3,size,size] float32 normalized,
                aux for unletterbox (scale,pad,orig_hw)
        """
        h0, w0 = img_bgr.shape[:2]
        sq, scale, pad = letterbox_square(img_bgr, self.img_size, pad_value=0)

        # BGR -> RGB -> float32 [0..1] -> normalize
        rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        chw = np.transpose(rgb, (2, 0, 1))  # HWC->CHW
        inp = np.expand_dims(chw, 0).astype(np.float32)  # [1,3,H,W]
        return inp, (scale, pad, (h0, w0))

    def infer(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Returns: prob mask (H,W) float32 in [0,1], same size as original image.
        """
        if not self._ok:
            raise RuntimeError("Model not loaded.")
        inp, meta = self.preprocess(img_bgr)
        scale, pad, orig_hw = meta

        if self.kind == "onnx":
            out = self.sess.run(None, {self.input_name: inp})[0]  # [1,1,S,S]
            prob_sq = 1.0 / (1.0 + np.exp(-out))  # sigmoid
            prob_sq = prob_sq[0, 0]               # (S,S)
        else:
            torch = self.torch
            x = torch.from_numpy(inp).to(self.device, non_blocking=True)
            with torch.inference_mode():
                logits = self.model(x)            # [1,1,S,S]
                prob_sq = torch.sigmoid(logits).squeeze(0).squeeze(0).detach().cpu().numpy()

        # map back to original size (remove pad, resize)
        prob = unletterbox(prob_sq.astype(np.float32), scale, pad, orig_hw)
        prob = np.clip(prob, 0.0, 1.0)
        return prob


# ----------------------------
# Qt Main Window
# ----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, img_size=DEFAULT_IMG_SIZE):
        super().__init__()
        self.setWindowTitle("U-Net Segmentation Viewer")
        self.resize(1300, 720)

        # Model
        self.segmenter = Segmenter(model_dir, img_size)

        # Top bar: Open button + model info
        self.open_btn = QtWidgets.QPushButton("Open Image…")
        self.open_btn.clicked.connect(self.on_open)

        self.model_label = QtWidgets.QLabel(self.segmenter.info())
        self.model_label.setStyleSheet("color:#777;")

        top_layout = QtWidgets.QHBoxLayout()
        top_layout.addWidget(self.open_btn)
        top_layout.addSpacing(12)
        top_layout.addWidget(self.model_label)
        top_layout.addStretch(1)

        # Three image panes
        self.view_orig = ScaledLabel()
        self.view_mask = ScaledLabel()
        self.view_overlay = ScaledLabel()

        for v in (self.view_orig, self.view_mask, self.view_overlay):
            v.setMinimumSize(200, 200)  # 可按需调整
            v.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(self._titled("Original", self.view_orig),   0, 0)
        grid.addWidget(self._titled("Mask",     self.view_mask),   0, 1)
        grid.addWidget(self._titled("Overlay",  self.view_overlay),0, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setHorizontalSpacing(8)

        # Log panel
        self.log = QtWidgets.QPlainTextEdit(readOnly=True)
        self.log.setMinimumHeight(140)
        self.log.setStyleSheet("font-family: Consolas, Menlo, monospace; font-size:12px;")

        # Central layout
        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)
        lay.addLayout(top_layout)
        lay.addLayout(grid, 1)
        lay.addWidget(self.log)
        self.setCentralWidget(central)

        if not self.segmenter.ok():
            self.append_log("[WARN] No ONNX or PTH model found in: {}".format(str(model_dir)))
            self.append_log("Put 'unet.onnx' or 'best.pth' under that folder.")

    def _titled(self, title: str, widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox(title)
        lay = QtWidgets.QVBoxLayout(box)
        lay.addWidget(widget)
        return box

    def append_log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"[{ts}] {text}")

    @QtCore.Slot()
    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose an image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        t0 = time.perf_counter()
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            self.append_log(f"Failed to read image: {path}")
            return

        # Show original
        self.view_orig.setPixmap(to_qpixmap_from_bgr(img_bgr))

        # If no model, just show info
        if not self.segmenter.ok():
            self.append_log("Model not loaded. Only showing original.")
            return

        # Inference
        t1 = time.perf_counter()
        prob = self.segmenter.infer(img_bgr)              # (H,W) float32 [0..1]
        t2 = time.perf_counter()

        # Mask grayscale preview
        mask_u8 = (np.clip(prob, 0, 1) * 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        self.view_mask.setPixmap(to_qpixmap_from_bgr(mask_bgr))

        # Overlay
        overlay = make_overlay(img_bgr, prob, color=(0, 0, 255), alpha=0.35)
        self.view_overlay.setPixmap(to_qpixmap_from_bgr(overlay))

        # Log
        self.append_log(f"Opened: {Path(path).name}  size={img_bgr.shape[1]}x{img_bgr.shape[0]}")
        self.append_log(f"Preprocess+I/O: {(t1 - t0)*1e3:.1f} ms  | Inference: {(t2 - t1)*1e3:.1f} ms  | Total: {(t2 - t0)*1e3:.1f} ms")
        self.append_log(f"Model: {self.segmenter.info()}  img_size={self.segmenter.img_size}")

def main():
    app = QtWidgets.QApplication([])
    w = MainWindow(model_dir=DEFAULT_MODEL_DIR, img_size=DEFAULT_IMG_SIZE)
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
