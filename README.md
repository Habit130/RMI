# RMI PlantSeg Server Delivery

这个仓库已经按 Linux 单卡 RTX 4090 目标迁移为 `PyTorch + ResNet101 + RMI` 的训练/验证版本，目标数据集固定为仓库同级目录 `../plantseg`。

## 当前交付范围

- 仅支持 `RMI + ResNet101`
- 仅支持 `plantseg`
- 文本输入固定使用 `caption[3]`
- 默认训练 `50 epoch`
- 测试阶段输出以下指标：
  - `IoU`
  - `Dice`
  - `Recall`
  - `mIoU`
  - `mACC`
- 测试阶段会把预测 mask 以与答案一致的 PNG 格式保存到配置指定目录

旧版 TensorFlow 代码仍保留在仓库中作为历史参考，但不再属于当前交付运行面。

## 环境

环境真相源是 `environment.yml`。

目标环境：

- Python 3.10
- PyTorch 2.1
- CUDA 11.8
- 单卡 RTX 4090

预训练模型使用 `torchvision` 官方 ResNet101 权重，首次训练时自动下载，不需要手动摆放权重文件。

## 数据约定

`plantseg` 目录必须与仓库同级，目录结构保持如下语义：

- `../plantseg/main.json`
- `../plantseg/images/*.jpg`
- `../plantseg/ann/*.png`

`main.json` 每条样本至少需要包含：

- `id`
- `image`
- `mask`
- `split`
- `caption`

其中 `caption[3]` 会被当作唯一训练/验证文本。

## 配置

默认配置文件是 `configs/plantseg_rmi_resnet.yaml`。

关键配置包括：

- 数据根目录
- batch 生成目录
- 词表输出路径
- 输入尺寸 `320`
- 文本步长 `64`
- 训练轮数 `50`
- checkpoint / metrics / 测试 mask 输出目录

## 使用流程

1. 创建 conda 环境
2. 构建 `plantseg` batch
3. 启动训练
4. 用最佳 checkpoint 跑测试并导出 mask

支持的命令：

```bash
conda env create -f environment.yml
conda activate rmi-plantseg
python build_batches.py --config configs/plantseg_rmi_resnet.yaml
python main.py --config configs/plantseg_rmi_resnet.yaml --mode train
python main.py --config configs/plantseg_rmi_resnet.yaml --mode test
```

训练与测试也会在 batch 不存在时自动构建 batch。

## 输出

默认输出目录为 `outputs/plantseg_rmi_resnet/`，包括：

- `checkpoint_best.pt`
- `checkpoint_last.pt`
- `metrics/*.json`
- `test_masks/*.png`

其中测试 mask 文件名与 `../plantseg/ann` 中的答案文件名一致，尺寸与像素格式保持一致。
