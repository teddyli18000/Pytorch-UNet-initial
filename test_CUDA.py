import torch
print("1. 显卡是否可用:", torch.cuda.is_available())
print("2. 当前显卡名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "识别失败")