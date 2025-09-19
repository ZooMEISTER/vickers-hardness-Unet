# infer_pth_gui.py  — U-Net(pth) 推理 + Tkinter 窗口显示
import os, sys, cv2, numpy as np, torch
from pathlib import Path
from tkinter import Tk, filedialog, Toplevel, Label, Button
from PIL import Image, ImageTk
import segmentation_models_pytorch as smp

WEIGHTS = r"runs/unet_r34_384/last.pth"
IMG_SIZE = 512
THRESH   = 0.5
ENCODER  = "resnet34"
ENCODER_WEIGHTS = None

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def letterbox_pad(img, size):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h*scale)), int(round(w*scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=img.dtype)
    canvas[:nh, :nw] = resized
    return canvas, scale, (size-nw, size-nh), (h, w)

def unpad_and_resize_mask(mask_sq, scale, pad, orig_hw):
    nh, nw = int(round(orig_hw[0]*scale)), int(round(orig_hw[1]*scale))
    mask_cropped = mask_sq[:nh, :nw]
    return cv2.resize(mask_cropped, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_NEAREST)

def build_model():
    return smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS,
                    in_channels=3, classes=1, activation=None)

def load_model(weights_path, device):
    model = build_model()
    # 兼容 PyTorch 2.5+ 的安全加载提示
    try:
        sd = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd)
    return model.to(device).eval()

def predict_mask(model, bgr, device):
    img_sq, scale, pad, orig_hw = letterbox_pad(bgr, IMG_SIZE)
    rgb = cv2.cvtColor(img_sq, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    rgb = (rgb - MEAN)/STD
    x = torch.from_numpy(np.transpose(rgb, (2,0,1))).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0,0].cpu().numpy()
    mask_sq = (prob >= THRESH).astype(np.uint8)*255
    return unpad_and_resize_mask(mask_sq, scale, pad, orig_hw)

def overlay(bgr, mask, color=(0,140,255), alpha=0.35):
    lay = np.zeros_like(bgr); lay[mask>0] = color
    return cv2.addWeighted(bgr, 1.0, lay, alpha, 0)

def compose_canvas(bgr, mask):
    vis = overlay(bgr, mask)
    m3  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    row = np.hstack([bgr, m3, vis])
    row = cv2.cvtColor(row, cv2.COLOR_BGR2RGB)  # Tkinter 用 RGB
    return Image.fromarray(row)

def choose_images():
    root = Tk(); root.withdraw()
    paths = filedialog.askopenfilenames(
        title="选择要识别的图片",
        filetypes=[("Images","*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff"), ("All Files","*.*")]
    )
    root.update(); root.destroy()
    return list(paths)

def show_image_in_window(pil_img, title="Result"):
    # 若图太大，适配屏幕宽度
    W_MAX = 1600
    w, h = pil_img.size
    if w > W_MAX:
        scale = W_MAX / w
        pil_img = pil_img.resize((int(w*scale), int(h*scale)), Image.BILINEAR)

    win = Toplevel()
    win.title(title)
    tk_img = ImageTk.PhotoImage(pil_img)
    lbl = Label(win, image=tk_img); lbl.image = tk_img  # 防止被GC
    lbl.pack()
    Button(win, text="关闭", command=win.destroy).pack()

def main():
    print(f"[INFO] device=", "cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = WEIGHTS if len(sys.argv) < 2 else sys.argv[1]
    if not Path(weights).exists():
        print(f"[ERR] 模型文件不存在：{weights}"); return
    print(f"[INFO] loading weights: {weights}")
    model = load_model(weights, device)

    img_paths = choose_images()
    if not img_paths:
        print("未选择图片，已退出。"); return

    # Tk 主窗口（作为弹窗父窗口）
    root = Tk(); root.withdraw()

    for p in img_paths:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[WARN] 读取失败：{p}"); continue
        mask = predict_mask(model, bgr, device)
        canvas = compose_canvas(bgr, mask)
        show_image_in_window(canvas, title=f"Result - {Path(p).name}")
        print(f"[OK] {p} 已显示。")

    root.mainloop()
    print("全部完成。")

if __name__ == "__main__":
    main()
