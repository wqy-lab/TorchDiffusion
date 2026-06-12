# Diffusion MNIST

## 环境配置

当前驱动支持 CUDA 12.9 时，请安装 CUDA 12.6 版 PyTorch（直接 `pip install torch` 会装 cu130，驱动不够新会导致 `cuda.is_available()` 为 `False`）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

若驱动已支持 CUDA 13.0+，可改用默认源：

```bash
pip install torch torchvision
pip install -r requirements.txt
```

## 运行训练

在 `config.py` 中切换数据集：

```python
dataset = 'cifar10'  # 或 'mnist'
```

```bash
python train.py
# 不指定 --init 时，使用 PyTorch 默认随机初始化

python train.py --init checkpoints/cifar10/2026-06-12_143005/best_model.pt
# 从已有权重继续训练（新时间戳目录，优化器重新初始化）

python train.py --init latest
# 从最近一次训练的 best_model.pt 初始化
```

也可在 `config.py` 中设置 `init_checkpoint = 'path/to/model.pt'`。

CIFAR-10 为 32×32 RGB，模型通道会自动设为 3。每次训练会按时间戳单独建目录，例如：

```
checkpoints/cifar10/2026-06-12_143005/
├── config.json      # 本次训练超参数
├── train.log        # 每个 epoch 的 loss
├── best_model.pt    # 最佳模型
├── last_model.pt    # 最后一轮模型
└── model_epoch_10.pt
```

`checkpoints/<dataset>/latest.txt` 会记录最近一次训练目录。

## 采样生成

训练完成后，从纯噪声逐步去噪生成图像：

```bash
python sample.py
# 默认读取 config.sample_run + config.sample_ckpt，无需手写模型路径
```

常用参数：

```bash
python sample.py -n 16
python sample.py -c last          # 用 last_model.pt
python sample.py -c 30            # 用 model_epoch_30.pt
python sample.py -r 2026-06-12_143005 -c best
python sample.py -s 8
# 导出时把 32x32 放大到 256x256，默认 sample_scale=8
```

也可在 `config.py` 中设置：

```python
sample_run = 'latest'   # 或指定目录名
sample_ckpt = 'best'      # best、last、30
```

采样流程：从 `x_T ~ N(0, I)` 出发，按 `t = T-1, ..., 0` 逐步调用 UNet 预测噪声，再用 `p_sample` 得到 `x_{t-1}`，最终得到 `x_0`。

## 文件结构

```
├── train.py              # 训练入口
├── sample.py             # 采样生成
├── config.py             # 超参数
├── utils/
│   └── run_dir.py        # 训练目录与日志管理
├── models/
│   ├── diffusion.py     # Diffusion 调度
│   └── unet_time.py      # 带 time embedding 的 UNet
├── checkpoints/          # 模型保存目录
└── requirements.txt
```
