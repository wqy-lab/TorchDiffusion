import torch

dataset = 'cifar10'  # 'mnist' or 'cifar10'

T = 200
beta_start = 1e-4
beta_end = 0.02
batch_size = 128
lr = 1e-3
epochs = 200
early_stopping_patience = 200
early_stopping_min_delta = 1e-5
base_channels = 64
channel_mults = [1, 2, 4, 8]
time_dim = 256
emb_size = 32
num_heads = 8
num_groups = 8
dropout = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET_PRESETS = {
    'mnist': {'image_size': 32, 'in_channels': 1, 'out_channels': 1},
    'cifar10': {'image_size': 32, 'in_channels': 3, 'out_channels': 3},
}

_preset = DATASET_PRESETS[dataset]
image_size = _preset['image_size']
in_channels = _preset['in_channels']
out_channels = _preset['out_channels']
checkpoint_base = f'checkpoints/{dataset}'
init_checkpoint = None  # 初始权重路径，None 表示 PyTorch 默认随机初始化
sample_run = 'latest'   # latest 或某次训练目录名，如 2026-06-12_180949
sample_ckpt = 'best'    # best、last，或 epoch 编号如 30
sample_scale = 8        # 导出时放大倍数，32x32 -> 256x256
sample_padding = 4      # 拼图间距（像素）

sample_w = 4