import numpy as np
import torch

# 加载 .npy 文件
#npy_array = np.load('/mnt/nfs/work/albertliu/RL2/actions.npy')
npy_array = np.load('/mnt/nfs/work/albertliu/RL2/observations.npy')
print(npy_array.shape)
#npy_array=npy_array.reshape(10,1000,11)
print(npy_array.shape)
# 将 NumPy 数组转换为 PyTorch 张量
torch_tensor = torch.from_numpy(npy_array)

# 保存为 .pt 文件
torch.save(torch_tensor, '/mnt/nfs/work/albertliu/RL2/expert_data/observations/HopperFH-v0_airl.pt')
#torch.save(torch_tensor, '/mnt/nfs/work/albertliu/RL2/expert_data/actions/HopperFH-v0_airl.pt')
