# From U-Net to TransNet Experiment Reproduction 3.0

本项目用于完成 U-Net 复现实验，并为后续向 TransUNet / TransNet 的迁移实验提供统一基线。当前版本已经在新的电脑、新的 Anaconda 环境和新的 VSCode 环境下完成完整跑通。

## 1. 项目目标

当前阶段的核心目标：

1. 跑通 U-Net 原始风格训练流程  
2. 在本地新环境上完成训练、验证/推理、阈值扫描全流程  
3. 固定一套可复现的目录结构、命令和输出规范  
4. 为后续 TransUNet / TransNet 复现实验保留统一接口  

---

## 2. 当前已完成内容

当前项目已完成或已验证通过的部分包括：

- 训练主程序：`src_improved/Train2_0.py`
- 数据读取与 patch 采样：`src_improved/DataLoader2_0.py`
- 验证 / 推理：`src_improved/Inference2_0.py`
- 阈值扫描：`src_improved/ThresholdSweep2_0.py`
- 新电脑本地环境配置完成
- 新 VSCode 工作区配置完成
- 训练数据路径与输出路径已改为当前机器可用路径
- 训练、推理、threshold sweep 已完成一次最终实验

---

## 3. 环境说明

### 3.1 硬件环境
- 设备：MacBook Pro
- 计算后端：Apple Silicon / MPS
- 开发工具：VSCode

### 3.2 软件环境
- Python: 3.11.15
- PyTorch: 2.11.0
- OpenCV: 4.13.0
- tifffile: 已安装
- ipykernel: 已安装
- jupyterlab: 已安装

### 3.3 Conda 环境
默认环境：

```bash
conda activate ragseg_torch
```

### 3.4 额外依赖说明
若读取 `.tif` 文件时报错：

```bash
ValueError: <COMPRESSION.LZW: 5> requires the 'imagecodecs' package
```

需要安装：

```bash
pip install imagecodecs
```

---

## 4. 项目目录结构

典型目录结构如下：

```bash
From U-Net to TransNet Experiment Reproduction 3.0/
├── README.md
├── processed_unet_train_auto_originalstyle/
├── processed_unet_test_auto_originalstyle/
├── src_improved/
│   ├── Train2_0.py
│   ├── DataLoader2_0.py
│   ├── Inference2_0.py
│   ├── ThresholdSweep2_0.py
│   └── ...
├── outputs_train_improved/
├── outputs_val_infer_improved/
├── outputs_threshold_sweep_improved/
└── train.log
```

各目录作用：

- `processed_unet_train_auto_originalstyle/`  
  训练 / 验证使用的预处理数据

- `processed_unet_test_auto_originalstyle/`  
  推理或测试使用的数据

- `src_improved/`  
  当前实验主代码目录

- `outputs_train_improved/`  
  训练输出目录，通常包括 checkpoint、日志、训练曲线等

- `outputs_val_infer_improved/`  
  验证 / 推理输出目录，通常包括概率图、预测 mask、推理 manifest

- `outputs_threshold_sweep_improved/`  
  threshold sweep 输出目录，通常包括 CSV、JSON 总结和最佳阈值结果

---

## 5. 当前关键路径

项目根目录：

```bash
/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0
```

训练数据目录：

```bash
/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/processed_unet_train_auto_originalstyle
```

测试 / 推理数据目录：

```bash
/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/processed_unet_test_auto_originalstyle
```

代码目录：

```bash
/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/src_improved
```

---

## 6. 训练配置

一次实际训练中使用的配置如下：

```json
{
  "processed_root": "/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/processed_unet_train_auto_originalstyle",
  "save_dir": "/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0/outputs_train_improved",
  "device": "mps",
  "epochs": 50,
  "batch_size": 1,
  "lr": 0.0001,
  "momentum": 0.99,
  "weight_decay": 0.0,
  "input_size": 572,
  "output_size": 388,
  "patches_per_image": 32,
  "elastic_deform": true,
  "displacement_std": 10.0,
  "grid_size": 3,
  "normalize": "zscore",
  "gray_value_aug": true,
  "use_bottleneck_dropout": true,
  "dropout_p": 0.5
}
```

说明：

- `device = mps`：使用 Mac 的 MPS 后端
- `batch_size = 1`：符合 U-Net 小样本分割实验常见设置
- `input_size = 572`, `output_size = 388`：对应原始 U-Net 风格裁剪逻辑
- `elastic_deform = true`：开启弹性形变增强
- `gray_value_aug = true`：开启灰度增强
- `use_bottleneck_dropout = true`：在 bottleneck 使用 dropout 抑制过拟合

---

## 7. 运行方法

### 7.1 进入项目目录

```bash
cd "/Users/brian/Desktop/VCL318/U-Net/From U-Net to TransNet Experiment Reproduction 3.0"
```

### 7.2 激活环境

```bash
conda activate ragseg_torch
```

### 7.3 训练

```bash
python -u src_improved/Train2_0.py 2>&1 | tee train.log
```

### 7.4 验证 / 推理
根据当前脚本配置运行推理：

```bash
python -u src_improved/Inference2_0.py
```

### 7.5 阈值扫描

```bash
python -u src_improved/ThresholdSweep2_0.py
```

---

## 8. 常见问题

### 8.1 MPS 的 pin_memory warning

若出现：

```bash
'pin_memory' argument is set as true but not supported on MPS now
```

这是 MPS 当前不支持 `pin_memory`，只是 warning，不会直接导致训练失败。

### 8.2 TIFF LZW 压缩报错

若出现：

```bash
ValueError: <COMPRESSION.LZW: 5> requires the 'imagecodecs' package
```

执行：

```bash
pip install imagecodecs
```

### 8.3 Matplotlib font cache 提示

若出现：

```bash
Matplotlib is building the font cache; this may take a moment.
```

这是首次建立字体缓存，不是错误。

### 8.4 训练 loss 很快变低是否等于效果好
不等于。  
`train loss` 很低只说明模型对训练集拟合很快，是否真正效果好需要看：

- validation loss
- validation dice / IoU
- 可视化预测结果
- 最优 checkpoint 对应 epoch

因此实验结果默认以验证集最佳模型为准，而不是只看最后一个 epoch。

---

## 9. 实验记录建议

后续继续实验时，建议固定保留以下内容：

1. 训练命令
2. 训练配置
3. 最优 checkpoint
4. 推理输出目录
5. threshold sweep 结果
6. 最终汇报时使用的核心图和指标

建议每次实验至少记录：

- 数据目录
- 训练轮数
- 学习率
- 是否使用增强
- 最佳阈值
- best epoch
- best val dice / IoU

---

## 10. 当前阶段结论

当前项目已经在新环境中完成完整跑通，说明：

- 路径迁移成功
- 环境配置基本稳定
- U-Net 训练、推理、threshold sweep 工作链路可用
- 可以在此基础上继续开展 TransUNet / TransNet 迁移实验

当前这版代码和目录可以视为后续迁移实验的稳定基线版本。

---

## 11. 后续建议

下一阶段建议工作：

1. 固定并整理当前 U-Net 最终基线结果
2. 补充 best checkpoint、best epoch、验证指标的统一记录
3. 在同一数据基础上设计 TransUNet 对照实验
4. 按统一格式完成汇报材料和实验文档

---

## 12. 五部分复现框架

后续所有复现、迁移、改进实验，统一按五部分组织：

1. `dataloader`
2. `model`
3. `train`
4. `load weight and test`
5. `output metrics / report`

这也是当前项目后续继续扩展时的默认组织方式。

---

## 13. 备注

本 README 对应的是本地最终跑通版本。  
若后续修改脚本名、路径、输出目录或加入新模型，请同步更新本文件。
