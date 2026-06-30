# MNIST Conditional DDIM

这是一个基于 MNIST 的条件 DDIM 数字生成项目。项目目标是用一个尽量清晰、可复现的小模型展示：如何把“指定类别”作为条件注入扩散模型，并通过 DDIM 采样从随机噪声生成对应的手写数字。

例如，输入：

```text
给出数字2
```

模型会生成若干张手写数字 `2` 的图片。也可以输入一组数字：

```text
612389
```

模型会按顺序生成一张数字组图。

## DDIM构建理解

扩散模型通常比普通分类或回归任务更抽象：训练时不是直接预测类别，而是学习如何从噪声中恢复数据分布。MNIST 数据集足够小、标签明确、视觉结果直观，因此很适合作为理解 DDPM / DDIM 条件生成的入门实验。

这个项目重点想说明两件事：

- MNIST 的数字标签可以作为生成条件，让模型学会“生成某个数字”。
- DDIM 可以用更少的采样步数完成生成，比完整 DDPM 采样更快。

## DDIM 构建思路

模型使用一个轻量级条件 U-Net 作为噪声预测网络。训练时，每张 MNIST 图片会被随机加噪，模型输入包括：

```text
加噪图片 x_t
时间步 t
数字标签 y
```

模型目标是预测加入图片中的噪声：

```text
model(x_t, t, y) -> predicted_noise
```

训练完成后，采样阶段从随机噪声开始，根据指定标签不断反推更干净的图像。这里使用 DDIM 采样，因此可以只取一部分时间步，例如用 `50` 步完成从噪声到图片的生成。

项目还加入了 classifier-free guidance：训练时会随机丢弃一部分标签，让模型同时学习“无条件生成”和“有条件生成”。采样时再用 `guidance_scale` 加强指定数字的生成方向。

## 文本条件如何处理

MNIST 本身没有自然语言描述，只有数字标签。因此这里没有训练复杂的文本编码器，而是使用一个轻量规则解析器：

```text
给出数字2 -> 2
give me digit two -> 2
612389 -> [6, 1, 2, 3, 8, 9]
```

也就是说，文本只负责转换成 MNIST 的类别条件，真正负责生成图片的是条件 U-Net + DDIM 采样器。

## 项目结构

```text
.
├── README.md
├── MNIST_DDIM_COMMANDS.md
└── DDPM&DDIM/
    ├── MNIST_DDIM_Text.py
    ├── data/
    └── mnist_ddim_runs/
        ├── checkpoints/
        ├── samples/
        └── outputs/
```

核心代码在：

```text
~/MNIST_DDIM_Text.py
```

更完整的命令记录见：

```text
MNIST_DDIM_COMMANDS.md
```

## 快速开始

安装依赖后，进入项目根目，训练模型：

```powershell
python ".~\MNIST_DDIM_Text.py" train
```

生成单个数字：

```powershell
python ".~\MNIST_DDIM_Text.py" sample --prompt "给出数字2" --num-images 8
```

生成数字组图：

```powershell
python ".~\MNIST_DDIM_Text.py" sample-sequence --sequence 612389
```

默认输出目录：

```text
~/mnist_ddim_runs/outputs/
```

> Windows PowerShell 中，路径 `DDPM&DDIM` 必须加引号，否则 `&` 会被解析成运算符。

## 常用参数

```text
--epochs          训练轮数，默认 50
--batch-size      batch 大小，默认 256
--timesteps       训练扩散步数，默认 1000
--sample-steps    DDIM 采样步数，默认 50
--eta             DDIM 随机性，默认 0
--guidance-scale  条件引导强度，默认 1.5
--device          训练/采样设备，默认 cuda
```

示例：

```powershell
python ".~\MNIST_DDIM_Text.py" train --epochs 20 --batch-size 256
```

```powershell
python ".~\MNIST_DDIM_Text.py" sample --prompt "给出数字2" --sample-steps 100
```

## 结果说明

训练过程中，脚本会定期保存预览图到：

```text
~/mnist_ddim_runs/samples/
```

每张预览图包含 `0-9` 的条件生成结果，可以用来观察模型从噪声到可辨认数字的学习过程。

用户主动生成的图片会保存到：

```text
~/mnist_ddim_runs/outputs/
```

## 项目意义

这个项目是一个小而完整的扩散模型实验：

- 能看到条件扩散模型的训练闭环。
- 能理解标签条件如何影响生成结果。
- 能比较 DDIM 少步采样带来的效率优势。
- 能用非常直观的 MNIST 图片验证“从噪声生成图像”的过程。

后续可以在这个基础上扩展到更复杂的数据集、更强的 U-Net、文本编码器，或者加入最近邻检索实验来分析模型是否存在记忆训练样本的问题。
