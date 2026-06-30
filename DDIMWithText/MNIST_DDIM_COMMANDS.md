# MNIST 条件 DDIM 常用命令

本文档记录 `DDIM/MNIST_DDIM_Text.py` 的常用训练、采样、GPU 检查和参数调整命令。

## 检查 CUDA 和显卡

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

持续观察 GPU 使用情况：

```powershell
nvidia-smi -l 1
```

训练启动时，脚本会打印类似信息：

```text
training device: cuda
cuda device: NVIDIA GeForce RTX 5060 Ti
model parameter device: cuda:0
data loader workers: 4
```

看到 `model parameter device: cuda:0` 就说明模型参数已经在 GPU 上。

## 正式训练

使用默认参数训练：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train
```

当前默认值包括：

```text
epochs = 50
batch_size = 256
device = cuda
timesteps = 1000
sample_steps = 50
lr = 2e-4
guidance_scale = 1.5
drop_label_prob = 0.1
num_workers = 4
```

指定训练轮数：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 20
```

指定 batch size：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 20 --batch-size 256
```

如果显存足够，可以尝试更大的 batch：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 20 --batch-size 512
```

如果出现 CUDA out of memory，就降低 batch size：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 20 --batch-size 128
```

## 减少预览图片保存频率

默认每个 epoch 会生成一张预览图：

```text
DDIM/mnist_ddim_runs/samples/epoch_001.png
DDIM/mnist_ddim_runs/samples/epoch_002.png
...
```

每 5 个 epoch 保存一次：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 50 --sample-every 5
```

完全关闭训练过程中的预览图：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 50 --sample-every 0
```

## 从 checkpoint 继续训练

默认 checkpoint 保存位置：

```text
DDIM/mnist_ddim_runs/checkpoints/latest.pt
```

继续训练：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --resume ".\DDIM\mnist_ddim_runs\checkpoints\latest.pt" --epochs 100
```

注意：`--epochs 100` 表示训练到第 100 个 epoch，不是额外再训练 100 个 epoch。

## 生成数字图片

训练完成后，输入中文提示生成数字 2：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample --prompt "给出数字2" --num-images 8
```

英文提示也可以：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample --prompt "give me digit two" --num-images 8
```

生成结果默认保存到：

```text
DDIM/mnist_ddim_runs/outputs/prompt_digit_2.png
```

指定 checkpoint：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample --prompt "给出数字2" --checkpoint ".\DDIM\mnist_ddim_runs\checkpoints\latest.pt" --num-images 8
```

指定输出目录：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample --prompt "给出数字2" --output-dir ".\DDIM\my_outputs" --num-images 8
```

## 按顺序生成一组数字

例如生成 `612389` 这组数字：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample-sequence --sequence 612389
```

它会把字符串解析成：

```text
[6, 1, 2, 3, 8, 9]
```

然后按从左到右的顺序生成一张横向组图。

默认输出位置：

```text
DDIM/mnist_ddim_runs/outputs/sequence_612389.png
```

指定 checkpoint：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample-sequence --sequence 612389 --checkpoint ".\DDIM\mnist_ddim_runs\checkpoints\latest.pt"
```

调整数字之间的间距：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample-sequence --sequence 612389 --padding 4
```

使用更多 DDIM 采样步数：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample-sequence --sequence 612389 --sample-steps 100
```

## DDIM 采样参数

采样步数：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample --prompt "给出数字2" --sample-steps 50
```

常见选择：

```text
20  步：速度快，质量可能差一点
50  步：默认值，速度和质量比较平衡
100 步：更稳，但更慢
```

控制随机性：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample --prompt "给出数字2" --eta 0
```

`eta = 0` 是确定性 DDIM；`eta > 0` 会增加随机性。

控制条件强度：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample --prompt "给出数字2" --guidance-scale 1.5
```

常见选择：

```text
1.0：不额外加强条件
1.5：默认值，比较推荐
2.0：更强调指定数字，但可能更僵硬
3.0+：可能出现伪影
```

## 常用调参建议

提高生成质量：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 50 --batch-size 256 --lr 2e-4
```

loss 抖动明显时：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 50 --batch-size 128 --lr 1e-4
```

显存足够，希望模型更强一点：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 50 --batch-size 256 --base-channels 96
```

只想更快检查流程：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 1 --batch-size 8 --base-channels 16 --timesteps 50 --sample-steps 5 --limit-train-batches 1 --log-every 1 --output-dir ".\DDIM\mnist_ddim_smoke"
```

## 常见问题

CPU 利用率高但 GPU 看起来不动：

先用下面命令确认 GPU：

```powershell
nvidia-smi -l 1
```

MNIST 很小，CPU 会负责数据加载、图片保存和 Python 调度；GPU 可能是短时间突增，不一定在任务管理器默认图表里明显显示。

如果想减少 CPU 和图片保存干扰：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train --epochs 50 --sample-every 5 --num-workers 2
```

找不到 checkpoint：

说明还没有正式训练，先运行：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" train
```

然后再采样：

```powershell
python ".\DDIM\MNIST_DDIM_Text.py" sample --prompt "给出数字2" --num-images 8
```

