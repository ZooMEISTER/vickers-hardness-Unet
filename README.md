### U-Net + OpenCV 识别维氏压痕

```
unet-vickers
├─ data		# 数据集
│   ├─ images	# 原图	
│   └─ masks	# 标注图
├─ runs
│   └─ unet_r34_512		# 训练结果
│        ├─ best.pth	# 最好的
│        ├─ curves.png	# 指标曲线
│        ├─ curves_0.png
│        ├─ history.json	# 每一epoch指标
│        ├─ history_0.json
│        ├─ last.pth	# 最后的
│        └─ unet.onnx	# onnx格式
├─ gpu_check.py		# 检查设备 GPU
├─ infer_pth_gui.py		# 打开一张图片并分析
├─ plot_history.py		# 生成curve.png
├─ train.py		# 训练
└─ ui_infer.py		# 可视化 UI 界面
```

#### 训练评估

![训练评估](runs\unet_r34_512\curves.png)

#### 运行

![runtime_1](runs\imgs\ui_1.jpg)

![runtime_2](runs\imgs\ui_2.jpg)

![runtime_3](runs\imgs\ui_3.jpg)

![runtime_4](runs\imgs\ui_4.jpg)

![runtime_5](runs\imgs\ui_5.jpg)

![runtime_6](runs\imgs\ui_6.jpg)
