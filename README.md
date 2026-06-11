# 石头样本混合检测（二分类）

基于 **MobileNetV3-Small** 的轻量化图像分类项目，用于判断石头照片是否为**混合样本**（混合了 B～E 多种尺寸）还是**单一尺寸样本**（A～E 中某一类）。

## 任务说明

根据 `石头样本/石头样本/说明文档.pdf`：

| 文件夹 | 含义 | 标签 |
|--------|------|------|
| A～E | 单一尺寸的石头图像，A 最大、E 最小 | `单一尺寸` (0) |
| 混合样本 | 混合了 B～E 类石头 | `混合样本` (1) |

当前数据集共 **27 张 HEIC 图像**（单一尺寸 22 张，混合 5 张）。

## 模型结构

采用 torchvision 提供的 **MobileNetV3-Small**，在 ImageNet 预训练权重基础上微调最后一层，用于二分类。

```
输入: RGB 图像 [3, 224, 224]
         │
         ▼
┌─────────────────────────────────────┐
│  features（特征提取骨干）              │
│  ├─ Stem: Conv3×3 + BN + Hardswish  │
│  └─ 12 × InvertedResidual 倒残差块    │
│       ├─ 1×1 扩展卷积                │
│       ├─ 3×3 / 5×5 深度可分离卷积     │
│       ├─ Squeeze-Excitation（部分块） │
│       └─ 1×1 投影卷积                │
│     通道变化: 16→24→40→48→96→576    │
└─────────────────────────────────────┘
         │
         ▼
  AdaptiveAvgPool2d(1)   →  [576]
         │
         ▼
┌─────────────────────────────────────┐
│  classifier（分类头）                 │
│  ├─ Linear(576 → 1024) + Hardswish  │
│  ├─ Dropout(0.2)                     │
│  └─ Linear(1024 → 2)  ← 替换原 1000 类 │
└─────────────────────────────────────┘
         │
         ▼
  Softmax → [P(单一尺寸), P(混合样本)]
```

### 关键参数

| 项目 | 数值 |
|------|------|
| 骨干网络 | MobileNetV3-Small |
| 参数量 | ~1.52M |
| 输入尺寸 | 224 × 224 |
| 输出类别 | 2（单一尺寸 / 混合样本） |
| 预训练 | ImageNet-1K (`IMAGENET1K_V1`) |
| 修改部分 | 仅替换 `classifier[-1]` 为 `Linear(1024, 2)` |

### 训练策略

- **迁移学习**：前 5 个 epoch 冻结骨干，仅训练分类头；之后解冻全网，学习率降为原来的 1/10
- **类别权重**：对混合样本（少数类）使用加权交叉熵损失
- **数据增强**：随机裁剪、水平/垂直翻转、颜色抖动
- **评估**：5 折分层交叉验证 + 全量数据训练最终模型

## 项目结构

```
stone_classifier/
├── README.md           # 本文件
├── README.pdf          # 项目说明 PDF（由 generate_readme_pdf.py 生成）
├── generate_readme_pdf.py  # README → PDF 生成脚本
├── requirements.txt    # Python 依赖
├── dataset.py          # 数据集定义与样本收集
├── model.py            # MobileNetV3-Small 构建
├── preprocess.py       # HEIC → 224×224 JPEG 缓存
├── train.py            # 训练脚本
├── inference.py        # 推理脚本
├── cache/              # 预处理图像缓存（自动生成）
└── output/             # 模型与结果输出
    ├── mobilenetv3_stone_binary.pt
    ├── training_report.json
    ├── inference_results.csv
    └── inference_results.json
```

## 环境安装

```powershell
cd stone_classifier
pip install -r requirements.txt
```

依赖：Python 3.10+、PyTorch、torchvision、pillow、pillow-heif、tqdm

> 首次训练会自动下载 ImageNet 预训练权重（约 10 MB）。若遇 SSL 证书问题，可手动下载至 `%USERPROFILE%\.cache\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth`。

## 使用方法

### 训练

```powershell
python train.py
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `../石头样本/石头样本` | 原始 HEIC 数据目录 |
| `--epochs` | 30 | 训练轮数 |
| `--folds` | 5 | 交叉验证折数 |
| `--batch-size` | 4 | 批大小 |
| `--lr` | 1e-3 | 初始学习率 |

### 推理

```powershell
python inference.py
```

对 `--data-dir` 下所有图像进行预测，结果写入 `output/inference_results.csv` 和 `.json`。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `output/mobilenetv3_stone_binary.pt` | 模型权重路径 |
| `--data-dir` | `../石头样本/石头样本` | 待推理图像目录 |

推理输出字段：`predicted_label`（预测类别）、`is_mixed_pred`（是否为混合）、`prob_mixed`（混合概率）、`confidence`（置信度）。

## 训练结果

在 27 张样本（单一尺寸 22 张、混合样本 5 张）上完成 5 折分层交叉验证与全量推理，结果如下：

| 指标 | 结果 |
|------|------|
| 5 折交叉验证平均准确率 | **92%** |
| 全量推理准确率 | **96.3%（26/27）** |
| 混合样本识别 | **4/5 正确** |
| 单一尺寸识别 | **22/22 正确** |

### 误判分析

| 文件名 | 真实标签 | 预测标签 | 混合概率 | 说明 |
|--------|----------|----------|----------|------|
| `IMG_9019.jpg` | 混合样本 | 单一尺寸 | 29.27% | 唯一误判，混合概率未超过 50% 阈值 |

### 各类别表现

- **单一尺寸（A～E）**：22 张全部预测正确，模型对各类单一尺寸样本区分稳定。
- **混合样本**：5 张中 4 张正确（`IMG_9018`、`IMG_9020`、`IMG_9021`、`IMG_9022`），1 张误判（`IMG_9019`）。

详细数值见 `output/training_report.json` 与 `output/inference_results.json`。

## 数据目录约定

```
石头样本/石头样本/
├── A/          # 单一尺寸（最大）
├── B/
├── C/
├── D/
├── E/          # 单一尺寸（最小）
├── 混合样本/    # 混合 B～E
└── 说明文档.pdf
```

程序通过文件夹名自动打标：**A～E 为负类，其余含图像的子文件夹为正类**（混合样本）。
