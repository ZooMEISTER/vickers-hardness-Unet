import torch
print(torch.cuda.is_available())  # 如果返回 True，说明 GPU 可用
print(torch.cuda.current_device())  # 输出当前设备的索引
print(torch.cuda.get_device_name(torch.cuda.current_device()))  # 输出当前 GPU 名称
