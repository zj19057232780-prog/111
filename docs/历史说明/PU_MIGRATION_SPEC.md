# IFMAML 原模型迁移至 PU 数据集 — 详细规格说明

> **文档目的**：结合可行性分析、源代码实际结构、原始需求，输出一份包含实现细节的完整规格文档。

---

# 1. 项目目标

在**不修改原有模型结构与训练流程**的前提下，实现：

- 使用 **PU 数据集**（.mat 格式单通道振动信号）替换原 Tsinghua 数据集
- 适配 PU 数据格式与组织结构
- 构建 few-shot meta tasks（5-way 1-shot / 5-way 5-shot）
- 支持跨工况迁移诊断（source → target）
- 保证原 MAML 训练流程不变

---

# 2. 源代码实际结构

> 需求文档假设的文件名与实际代码存在偏差，以下为实际项目文件与模块对应关系：

| 需求文档假设 | 实际文件 | 实际内容 |
|---|---|---|
| `model.py` | `maml_model.py` | Net4CNN + ConvBlock + CNN4Backbone + SEAttention |
| `maml.py` | 嵌入于 `maml_train_pic.py` | `MAML_learner` 类（train/test/fast_adapt 方法） |
| `train.py` | `maml_train_pic.py` | 训练入口 + 完整 MAML 内外循环 |
| `dataset.py` | `Datasets/pic_data.py` | `MAML_Dataset` + `Data_PU` 类 |
| `spca.py` | **不存在**，实现在 `normalize255.py` | `sparse_coding()` + `normalize()` |
| `fusion.py` | **不存在**，实现在 `Tsinghua.py` | `np.dstack` + `cv2.transpose` RGB 图像生成 |
| 路径配置 | `Datasets/Tsinghua_path_10C.py` | `T0`, `T3`, `T_valid` 路径常量 |

---

# 3. 现有关键代码解析

---

## 3.1 模型结构 — [maml_model.py](file:///c:/Users/JieZhang/Desktop/new-ifmaml/IFMAML-master/maml_model.py)

```python
# 模型层次结构
Net4CNN(
    features = CNN4Backbone(          # ConvBase 子类
        ConvBlock(channels=3→hidden, kernel=3, max_pool_factor=1)  # 4层
        ConvBlock(hidden→hidden, ...) × 3
        → flatten(x.size(0), -1)      # 输出 embedding_size 维向量
    )
    classifier = Linear(embedding_size → output_size)
)
```

**关键参数**（在 `maml_train_pic.py` 中初始化）：
| 参数 | 值 | 说明 |
|---|---|---|
| `hidden_size` | 64 | 卷积通道数 |
| `layers` | 4 | ConvBlock 层数 |
| `channels` | 3 | 输入通道数（RGB） |
| `embedding_size` | `(sample_len // 2^layers) * hidden` | 默认 256//16*64 = 1024 |
| `output_size` | `ways` | 分类类别数（5 或 10） |
| `max_pool_factor` | `4 / layers = 1` | 每层 maxpool stride=1 |

**输入要求**：
```
Tensor shape: [batch_size, 3, 64, 64]
dtype:       float32
value range: [0, 1] 归一化
```

---

## 3.2 原数据预处理管线

### 管线全流程

```
.mat 三通道信号（振动 + 电流 + 扭矩）
  │
  ▼ Tsinghua.py
sparse_coding(信号.T, k=3, max_iter=100, lma=1, tol=1e-4)
  输出: [N, 3] 三个稀疏编码成分
  │
  ▼ Tsinghua.py
normalize() → 映射到 [0, 255]
随机截取长度为 4096 (64×64) 的片段
reshape(64, 64)
  │
  ▼ Tsinghua.py
np.dstack([img1, img2, img3]) + cv2.transpose
保存为 PNG 到磁盘
  │
  ▼ Datasets/pic_data.py
PIL.Image.open() → resize(64,64) → np.array → /255.0 → reshape(-1, 3, 64, 64)
  │
  ▼ MAML_Dataset (torch.utils.data.Dataset)
l2l.data.MetaDataset + l2l.data.TaskDataset + FusedNWaysKShots
  │
  ▼ maml_train_pic.py — MAML_learner.train()
Meta-training loop (inner + outer)
```

### sparse_coding 实现 — [normalize255.py](file:///c:/Users/JieZhang/Desktop/new-ifmaml/IFMAML-master/normalize255.py)

```python
def sparse_coding(x, k=3, max_iter=100, lma=1, tol=1e-4):
    """
    简易稀疏编码（代码注释标注为 #Proposed）
    输入: x [n_samples, n_features]
    输出: alpha [n_samples, k]  —— k个稀疏成分
    """
    x = scipy.stats.zscore(x)
    n, d = x.shape
    x = x - np.mean(x)
    w = np.random.randn(d, k)
    for i in range(max_iter):        # 每轮迭代100次
        alpha = np.dot(x, w)
        for j in range(k):
            # 字典更新公式
            w[:, j] = np.dot(x.T, np.dot(x, w_old[:, j])) / (
                reduce(np.dot, (w_old[:, j].T, x.T, x, w_old[:, j])) + lma
            )
        if err < tol: break
    return alpha
```

### normalize 实现

```python
def normalize(x):
    """线性映射到 [0, 255]"""
    y_max, y_min = 255, 0
    x_max, x_min = np.max(x), np.min(x)
    return np.around((y_max - y_min) * (x - x_min) / (x_max - x_min) + y_min)
```

---

## 3.3 MAML 训练流程 — [maml_train_pic.py](file:///c:/Users/JieZhang/Desktop/new-ifmaml/IFMAML-master/maml_train_pic.py)

### MAML_learner 关键参数

```python
# train() 方法中
meta_lr       = 0.005      # 外循环学习率
fast_lr       = 0.05       # 内循环学习率（adaptation lr）
Epochs        = 1000
meta_batch_size = 16
adaptation_steps = 1  (5-shot) / 3 (1-shot)
loss          = CrossEntropyLoss(reduction='mean')
optimizer     = Adam(maml.parameters(), meta_lr)
```

### 训练循环伪代码

```python
for ep in range(Epochs):
    opt.zero_grad()
    for _ in range(meta_batch_size):       # 16 tasks per epoch
        learner = maml.clone()             # clone meta-parameters
        task = train_tasks.sample()        # sample a meta-task
        error, acc = fast_adapt(task, learner, ...)
        error.backward()                   # accumulate meta-gradients
    for p in maml.parameters():
        p.grad.data.mul_(1.0 / meta_batch_size)
    opt.step()                             # meta-update
```

### fast_adapt 伪代码

```python
def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways):
    data, labels = batch
    # 按偶数/奇数索引划分 support/query
    adaptation_data = data[0::2]     # support set
    evaluation_data = data[1::2]     # query set
    # Inner loop
    for step in range(adaptation_steps):
        pred = learner(adaptation_data)
        train_error = loss(pred, adaptation_labels)
        learner.adapt(train_error)   # gradient step with fast_lr
    # Evaluation
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    return valid_error, valid_accuracy
```

---

## 3.4 Dataset 接口 — [Datasets/pic_data.py](file:///c:/Users/JieZhang/Desktop/new-ifmaml/IFMAML-master/Datasets/pic_data.py)

```python
class MAML_Dataset(data.Dataset):
    def __init__(self, mode, ways):
        # mode: 'train' / 'validation' / 'test'
        self.height = 64
        self.width = 64
        self.__getdata__(mode)

    def __getdata__(self, mode):
        # 从磁盘PNG图片加载
        # 输出 self.x: [N, 3, 64, 64]  float32 [0,1]
        #      self.y: [N]           int32

    def __getitem__(self, item):
        return self.x[item], self.y[item]   # (3,64,64) Tensor, scalar label

    def __len__(self):
        return len(self.x)
```

**与 learn2learn 集成方式**：

```python
dataset = l2l.data.MetaDataset(MAML_Dataset(mode='train', ways=5))
tasks = l2l.data.TaskDataset(dataset, task_transforms=[
    l2l.data.transforms.FusedNWaysKShots(dataset, n_way, 2*shots),
    l2l.data.transforms.LoadData(dataset),
    l2l.data.transforms.RemapLabels(dataset, shuffle=True),
    l2l.data.transforms.ConsecutiveLabels(dataset),
], num_tasks=1000)
```

**Task 采样输出**：`(data, labels)` — data shape `[n_way * 2*shots, 3, 64, 64]`，其中偶数索引=支撑集，奇数索引=查询集。

---

# 4. PU 数据适配方案

---

## 4.1 数据来源与格式

| 属性 | 说明 |
|---|---|
| 格式 | `.mat` 文件（MATLAB v5 及以上） |
| 信号类型 | 单通道振动信号 |
| 工况划分 | 不同转速/负载组合为不同工况 |
| 故障类别 | 每个文件代表某类故障（如内圈故障、外圈故障等） |
| 读取方式 | `scipy.io.loadmat()` |

---

## 4.2 整体管线设计

```
PU .mat 文件
  │
  ▼ pu_loader.py
读取信号，按工况组织
  │
  ▼ pu_preprocess.py
滑窗分割（window=4096, overlap可配）
归一化 + 三通道构造（原始/包络/滤波）
  │
  ▼ 离线预处理阶段（pu_preprocess.py 调用 Tsinghua.py 逻辑）
sparse_coding() → normalize255 → reshape(64,64) → RGB融合 → 保存 PNG
      或
on-the-fly 阶段（在 Dataset 初始化时动态生成）
  │
  ▼ pu_dataset.py (PUMetaDataset → 内部 MAML_Dataset 兼容接口)
加载每个类别的样本，划分 source/target
  │
  ▼ maml_train_pic.py (不修改)
→ l2l MetaDataset → TaskDataset
  │
  ▼ MAML inner/outer loop (不修改)
→ Fast adaptation → 跨工况诊断
```

**推荐**：采用**离线预处理**，先调用 `pu_preprocess.py` + 复用 Tsinghua.py 中的 sparse_coding / normalize / RGB fusion 逻辑，生成 PNG 到磁盘。然后 `pu_dataset.py` 只负责从磁盘加载图片，与现有 `MAML_Dataset` 接口完全一致。原因是 `sparse_coding` 每次迭代100次，on-the-fly 会严重拖慢训练速度。

---

## 4.3 信号预处理

### 4.3.1 滑动窗口分割

```python
def sliding_window(signal, window_size=4096, overlap=0.5):
    """
    输入: signal [N,]  一维振动信号, float64
    输出: segments [M, window_size]  float64
    """
    step = int(window_size * (1 - overlap))
    segments = []
    for start in range(0, len(signal) - window_size + 1, step):
        segments.append(signal[start:start + window_size])
    return np.array(segments)
```

### 4.3.2 归一化

```python
def normalize(signal):
    """Z-score 标准化，或 Min-Max 归一化到 [0,1]"""
    return (signal - np.mean(signal)) / np.std(signal)
```

### 4.3.3 单通道 → 三通道扩展

PU 数据是单通道振动信号，但原模型要求 3 通道输入。构造方式：

```python
from scipy.signal import hilbert

def build_three_channel(signal_segment):
    """
    输入: signal_segment [4096,]  单通道振动信号
    输出: three_channel [4096, 3]  三通道信号
    """
    X1 = signal_segment                          # 原始振动信号
    X2 = np.abs(hilbert(signal_segment))         # Hilbert包络信号
    X3 = signal_segment                          # 复制原信号（或用带通滤波信号）
    return np.column_stack([X1, X2, X3])
```

**说明**：包络信号通过 Hilbert 变换取绝对值获得，反映信号的瞬时幅值。第三通道简单复制原信号，也可替换为带通滤波信号（如 500-2000Hz），取决于 PU 数据的采样频率。

### 4.3.4 稀疏编码 → RGB 图像生成

```python
def signal_to_image(three_channel_signal):
    """
    输入: three_channel_signal [4096, 3]
    输出: img [3, 64, 64]  float32 [0, 1]
    """
    from normalize255 import sparse_coding, normalize

    # 1) 稀疏编码
    alpha = sparse_coding(three_channel_signal, k=3, max_iter=100, lma=1, tol=1e-4)
    # alpha: [4096, 3]

    # 2) 归一化到 [0, 255] 并截取 4096 点
    img_points = 64 * 64  # 4096
    ch1 = normalize(alpha[:img_points, 0])
    ch2 = normalize(alpha[:img_points, 1])
    ch3 = normalize(alpha[:img_points, 2])

    # 3) reshape → 单通道图像
    img1 = np.reshape(ch1, (64, 64))
    img2 = np.reshape(ch2, (64, 64))
    img3 = np.reshape(ch3, (64, 64))

    # 4) RGB 融合
    img_rgb = np.dstack((img1, img2, img3))          # [64, 64, 3]
    img_rgb = cv2.transpose(img_rgb)                  # 转置对齐
    img_rgb = img_rgb / 255.0                        # 归一化到 [0, 1]
    return img_rgb.astype(np.float32)
```

**注意**：`sparse_coding` 调用了原 `normalize255.py` 中的函数，**不需要新建 `spca.py`**。如果需要独立文件封装以符合需求文档格式，可以创建一个 `spca.py` 作为 wrapper，直接调用 `normalize255.sparse_coding`。

---

# 5. 新增文件详细规格

---

## 5.1 `pu_loader.py` — PU 数据读取器

```python
import os
import scipy.io as scio
import numpy as np


class PULoader:
    """
    PU 数据集 .mat 文件读取器

    Attributes
    ----------
    root_path : str
        PU 数据集根目录
    file_map : dict
        工况名 → 故障类别列表 → 文件路径
    """

    def __init__(self, root_path):
        """
        Parameters
        ----------
        root_path : str
            数据根目录，目录结构示例:
            root_path/
              ├── condition_A/
              │     ├── fault_1.mat
              │     ├── fault_2.mat
              │     └── ...
              ├── condition_B/
              │     ├── fault_1.mat
              │     └── ...
              └── ...
        """
        self.root_path = root_path
        self.file_map = {}
        self._scan_files()

    def _scan_files(self):
        """扫描目录，建立工况→故障类→文件路径的映射"""
        for condition in os.listdir(self.root_path):
            cond_path = os.path.join(self.root_path, condition)
            if not os.path.isdir(cond_path):
                continue
            self.file_map[condition] = {}
            for fname in os.listdir(cond_path):
                if fname.endswith('.mat'):
                    fault_name = fname.replace('.mat', '')
                    self.file_map[condition][fault_name] = os.path.join(cond_path, fname)

    def load_file(self, file_path):
        """
        加载单个 .mat 文件

        Parameters
        ----------
        file_path : str

        Returns
        -------
        mat_dict : dict
            scipy.io.loadmat 返回的字典
        """
        return scio.loadmat(file_path)

    def get_signal(self, mat_dict):
        """
        从 mat 字典中提取振动信号

        自动检测信号对应的 key（常见 key: 'vibration', 'sig', 'data', 或第一个非系统 key）

        Parameters
        ----------
        mat_dict : dict

        Returns
        -------
        signal : np.ndarray [N,]  float64
        """
        for key in mat_dict.keys():
            if not key.startswith('__'):
                signal = mat_dict[key]
                if signal.ndim == 2 and signal.shape[1] == 1:
                    return signal.flatten()
                elif signal.ndim == 1:
                    return signal
                elif signal.ndim == 2:
                    return signal[:, 0]  # 取第一列
        raise KeyError('无法识别的 .mat 文件信号格式')

    def load_condition(self, condition_name):
        """
        加载某个工况下的全部故障类别信号

        Parameters
        ----------
        condition_name : str
            工况名称

        Returns
        -------
        fault_signals : dict
            fault_name → np.ndarray [N,] 一维振动信号
        """
        fault_signals = {}
        for fault_name, file_path in self.file_map[condition_name].items():
            mat_dict = self.load_file(file_path)
            fault_signals[fault_name] = self.get_signal(mat_dict)
        return fault_signals

    def get_condition_names(self):
        """返回所有工况名称列表"""
        return list(self.file_map.keys())

    def get_fault_names(self, condition_name):
        """返回某工况下所有故障类别名称列表"""
        return list(self.file_map[condition_name].keys())
```

---

## 5.2 `pu_preprocess.py` — 预处理（信号→图像）

```python
import numpy as np
from scipy.signal import hilbert
from normalize255 import sparse_coding, normalize
import cv2


def sliding_window(signal, window_size=4096, overlap=0.5):
    """
    滑动窗口分割

    Parameters
    ----------
    signal : np.ndarray [N,]
    window_size : int, default 4096
    overlap : float, [0, 1), default 0.5

    Returns
    -------
    segments : np.ndarray [M, window_size]
    """
    step = int(window_size * (1 - overlap))
    if step <= 0:
        step = 1
    n_windows = (len(signal) - window_size) // step + 1
    segments = np.zeros((n_windows, window_size), dtype=np.float64)
    for i in range(n_windows):
        start = i * step
        segments[i] = signal[start:start + window_size]
    return segments


def signal_normalize(signal):
    """
    Z-score 标准化

    Parameters
    ----------
    signal : np.ndarray

    Returns
    -------
    np.ndarray
    """
    return (signal - np.mean(signal, axis=-1, keepdims=True)) / (
        np.std(signal, axis=-1, keepdims=True) + 1e-10
    )


def build_three_channel(signal_segment):
    """
    单通道振动信号 → 三通道

    channel 1: 原始信号
    channel 2: Hilbert 包络信号
    channel 3: 原始信号（复本）

    Parameters
    ----------
    signal_segment : np.ndarray [L,]  L = window_size (e.g. 4096)

    Returns
    -------
    three_ch : np.ndarray [L, 3]
    """
    X1 = signal_segment
    X2 = np.abs(hilbert(signal_segment))
    X3 = signal_segment
    return np.column_stack([X1, X2, X3])


def signal_to_image(three_channel_signal, img_size=64):
    """
    三通道信号 → sparse_coding → normalize255 → reshape → RGB 图像

    调用原 normalize255.py 中的 sparse_coding 和 normalize 函数，
    保持与原 Tsinghua.py 完全一致的图像生成逻辑。

    Parameters
    ----------
    three_channel_signal : np.ndarray [L, 3]
    img_size : int, 64

    Returns
    -------
    image : np.ndarray [img_size, img_size, 3]  float32 [0, 255]
        原始 [0,255] 图像（归一化由 Dataset 加载时做 /255.0）
    """
    n_points = img_size * img_size  # 4096

    alpha = sparse_coding(three_channel_signal, k=3, max_iter=100, lma=1, tol=1e-4)
    # alpha: [L, 3]

    ch1 = normalize(alpha[:n_points, 0])
    ch2 = normalize(alpha[:n_points, 1])
    ch3 = normalize(alpha[:n_points, 2])

    img1 = np.reshape(ch1, (img_size, img_size))
    img2 = np.reshape(ch2, (img_size, img_size))
    img3 = np.reshape(ch3, (img_size, img_size))

    img_rgb = np.dstack((img1, img2, img3))  # [64, 64, 3]
    img_rgb = cv2.transpose(img_rgb)
    return img_rgb


def save_images(images, output_dir, class_name):
    """
    将图片保存到磁盘

    Parameters
    ----------
    images : np.ndarray [N, 64, 64, 3]  uint8 [0, 255]
    output_dir : str
    class_name : str
    """
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    for i, img in enumerate(images):
        save_path = os.path.join(class_dir, f'{class_name}_{i}.png')
        cv2.imwrite(save_path, img)
```

---

## 5.3 `pu_dataset.py` — Meta Task 数据集（替换原 `pic_data.py`）

```python
import os
import numpy as np
from torch.utils import data
from PIL import Image
from .pu_loader import PULoader
from .pu_preprocess import (
    sliding_window, signal_normalize, build_three_channel,
    signal_to_image,
)


def read_directory(directory_name, height, width):
    """
    从磁盘加载 PNG 图片

    保持与原 pic_data.py 中 read_directory() 完全一致的接口和行为。

    Parameters
    ----------
    directory_name : str
    height : int (64)
    width : int (64)

    Returns
    -------
    data : np.ndarray [-1, 3, height, width]  float32 [0, 1]
    """
    file_list = os.listdir(directory_name)
    img = []
    for each_file in file_list:
        img0 = Image.open(os.path.join(directory_name, each_file))
        gray = img0.resize((height, width))
        img.append(np.array(gray).astype(np.float))
    data = np.array(img) / 255.0
    data = data.reshape(-1, 3, height, width)
    return data


class PUMetaDataset(data.Dataset):
    """
    替换原 MAML_Dataset，保持完全相同的 __getitem__ / __len__ 接口

    Parameters
    ----------
    root_path : str
        PU 数据根目录（已预处理好 PNG 图片的目录，目录结构：root_path/工况名/故障类名/*.png）
    source_condition : str
        源工况名称
    target_condition : str
        目标工况名称
    img_size : int, default 64
    """

    def __init__(self,
                 root_path,
                 source_condition,
                 target_condition,
                 img_size=64):
        super().__init__()
        self.root_path = root_path
        self.source_condition = source_condition
        self.target_condition = target_condition
        self.height = img_size
        self.width = img_size

    def prepare_data(self):
        """
        预加载数据，遍历工况目录下各类别 PNG 图片
        """
        source_dir = os.path.join(self.root_path, self.source_condition)
        target_dir = os.path.join(self.root_path, self.target_condition)

        self.source_classes = sorted(os.listdir(source_dir))
        self.target_classes = sorted(os.listdir(target_dir))

        # 确保 source 和 target 的类别一致
        assert self.source_classes == self.target_classes, (
            f"Source 和 Target 类别不一致!\n"
            f"  Source: {self.source_classes}\n"
            f"  Target: {self.target_classes}"
        )

        self.num_classes = len(self.source_classes)

        # 加载各类别图片
        self.source_data = {}   # class_name → np.ndarray [n, 3, 64, 64]
        self.target_data = {}

        for cls_name in self.source_classes:
            src_cls_dir = os.path.join(source_dir, cls_name)
            tgt_cls_dir = os.path.join(target_dir, cls_name)

            self.source_data[cls_name] = read_directory(
                src_cls_dir, self.height, self.width
            )
            self.target_data[cls_name] = read_directory(
                tgt_cls_dir, self.height, self.width
            )

    def build_meta_tasks(self, n_way=5, k_shot=1, q_query=15, mode='train'):
        """
        构建 meta-task 所需的数据矩阵

        Parameters
        ----------
        n_way : int, 5
        k_shot : int, 1 或 5
        q_query : int, 15
        mode : str, 'train' → 使用 source_data, 'test' → 使用 target_data

        Returns
        -------
        data : np.ndarray [n_way, k_shot+q_query, 3, 64, 64]
        labels : np.ndarray [n_way, k_shot+q_query]
        """
        data_source = self.source_data if mode == 'train' else self.target_data
        class_names = list(data_source.keys())

        n_per_class = k_shot + q_query

        data_list = []
        label_list = []
        for cls_idx, cls_name in enumerate(class_names[:n_way]):
            cls_data = data_source[cls_name]
            indices = np.random.choice(len(cls_data), n_per_class, replace=False)
            data_list.append(cls_data[indices])
            label_list.append(np.full(n_per_class, cls_idx, dtype=np.int32))

        return np.stack(data_list, axis=0), np.stack(label_list, axis=0)

    def to_maml_dataset_format(self, mode='train'):
        """
        转换为与 MAML_Dataset 完全一致的接口格式

        Parameters
        ----------
        mode : str, 'train' / 'validation' / 'test'

        Returns
        -------
        x : np.ndarray [N, 3, 64, 64]
        y : np.ndarray [N]
        """
        data_source = self.source_data if mode == 'train' else self.target_data
        all_x = []
        all_y = []
        for cls_idx, (cls_name, cls_data) in enumerate(data_source.items()):
            all_x.append(cls_data)
            all_y.append(np.full(len(cls_data), cls_idx, dtype=np.int32))

        x = np.concatenate(all_x, axis=0)
        y = np.concatenate(all_y, axis=0)

        # 与原 MAML_Dataset 一致的 shuffle
        indices = np.random.permutation(len(x))
        return x[indices], y[indices]

    def __getitem__(self, item):
        """单样本访问 — 用于 learn2learn MetaDataset"""
        return torch.from_numpy(self._x[item]), self._y[item]

    def __len__(self):
        return len(self._x)

    def set_mode(self, mode):
        """
        设置当前数据集模式并填充 _x, _y

        Parameters
        ----------
        mode : str, 'train' / 'validation' / 'test'
        """
        self._x, self._y = self.to_maml_dataset_format(mode)
```

---

# 6. maml_train_pic.py 修改需求

## 6.1 数据加载替换

**原代码**（`maml_train_pic.py` 中的 `build_tasks` 方法）：
```python
from Datasets.Tsinghua_path_10C import T0, T3
dataset = l2l.data.MetaDataset(MAML_Dataset(mode=mode, ways=ways))
```

**修改后**：
```python
from pu_dataset import PUMetaDataset

# 在 main 或类初始化时：
pu_dataset = PUMetaDataset(
    root_path='./pu_data_processed',       # 预处理后的图片目录
    source_condition='N09_M07_F10',         # 源工况
    target_condition='N15_M01_F10',         # 目标工况
    img_size=64
)
pu_dataset.prepare_data()

# build_tasks 中：
dataset_obj = PUMetaDataset(
    root_path='./pu_data_processed',
    source_condition='N09_M07_F10',
    target_condition='N15_M01_F10',
    img_size=64
)
dataset_obj.prepare_data()
dataset_obj.set_mode(mode)  # 'train'/'validation'/'test'
dataset = l2l.data.MetaDataset(dataset_obj)
```

## 6.2 新增参数配置

在 `maml_train_pic.py` 的 `if __name__ == '__main__'` 块中新增：

```python
PU_CONFIG = {
    "root_path": "./pu_data_processed",
    "source_condition": "N09_M07_F10",
    "target_condition": "N15_M01_F10",
    "img_size": 64,
    "n_way": 5,
    "k_shot": 1,        # 或 5
    "q_query": 15,
}
```

## 6.3 训练流程 — 不修改

以下部分**完全不变**：
- MAML_learner 类的 `train()` / `test()` / `fast_adapt()` 方法
- `build_tasks()` 中的 L2L transforms（FusedNWaysKShots / RemapLabels / ConsecutiveLabels）
- 优化器 Adam(meta_lr=0.005)
- 内循环 fast_lr=0.05
- 损失函数 CrossEntropyLoss
- Net4CNN 模型初始化参数

---

# 7. 配置文件 — `config_pu.py`

```python
"""
PU 数据集训练配置文件
"""

PU_CONFIG = {
    # 数据
    "root_path": "./pu_data_processed",
    "source_condition": "N09_M07_F10",
    "target_condition": "N15_M01_F10",

    # 信号预处理
    "window_size": 4096,
    "overlap": 0.5,
    "img_size": 64,

    # Meta-learning
    "n_way": 5,
    "k_shot": 1,
    "q_query": 15,

    "inner_lr": 0.05,
    "outer_lr": 0.005,
    "epochs": 250,

    # MAML
    "meta_batch_size": 16,
    "adaptation_steps": {
        1: 3,   # 1-shot: 3 inner steps
        5: 1,   # 5-shot: 1 inner step
    },
}
```

---

# 8. 修改范围总结

| 文件 | 操作 | 说明 |
|---|---|---|
| `maml_model.py` | ❌ 不修改 | Net4CNN + SEAttention + ConvBlock 保持不变 |
| `maml_train_pic.py` | ✅ 轻微修改 | 替换数据源导入 + 新增 PU_CONFIG |
| `Datasets/pic_data.py` | ✅ 保留备用 | 原 Tsinghua 数据集兼容 |
| `normalize255.py` | ❌ 不修改 | sparse_coding + normalize 被新文件调用 |
| `Tsinghua.py` | ❌ 不修改 | 离线预处理脚本，可被复用 |
| `Datasets/Tsinghua_path_10C.py` | ❌ 不修改 | 原路径配置 |
| `my_utils/train_utils.py` | ❌ 不修改 | accuracy + MMD_loss |
| `my_utils/init_utils.py` | ❌ 不修改 | seed + sample_label_shuffle |
| **`pu_loader.py`** | ✅ 新增 | PU .mat 文件读取 |
| **`pu_preprocess.py`** | ✅ 新增 | 滑窗/归一化/三通道扩展/信号→图像 |
| **`pu_dataset.py`** | ✅ 新增 | PUMetaDataset，兼容 learn2learn |
| **`config_pu.py`** | ✅ 新增 | PU 训练参数配置 |

---

# 9. 验收标准

| 编号 | 标准 | 验证方式 |
|---|---|---|
| 1 | 程序可正常运行，无 import 错误 | `python maml_train_pic.py` 进入训练循环 |
| 2 | 支持跨工况训练（source→target） | train 使用 source_condition，test 使用 target_condition |
| 3 | 支持 1-shot 与 5-shot | 修改 `PU_CONFIG["k_shot"]` 后运行正常 |
| 4 | 输入 shape 为 `[batch_size, 3, 64, 64]` | 在 `fast_adapt` 中打印 `data.shape` 验证 |
| 5 | 数据 float32，归一化到 [0,1] | 在 `__getitem__` 返回值检查 dtype 和 range |
| 6 | 标签为 LongTensor | learn2learn RemapLabels transform 自动转换 |
| 7 | 不修改 maml_model.py | diff 验证 |
| 8 | MAML 内/外循环逻辑不变 | diff 验证 maml_train_pic.py 中的训练逻辑 |
| 9 | 输出诊断准确率 | 终端打印 `Meta Train Accuracy` / `Meta Valid Accuracy` |

---

# 10. 风险与注意事项

| 风险 | 缓解措施 |
|---|---|
| PU 数据 .mat 文件内部 key 名不确定 | `pu_loader.get_signal()` 自动检测第一个非系统 key |
| sparse_coding 计算耗时大 | 推荐离线预处理生成 PNG，仅训练阶段从磁盘加载 |
| Source/Target 类别数不一致 | `pu_dataset.prepare_data()` 中 assert 检查 |
| 包络信号可能噪声大 | 必要时先对原始信号做带通滤波再取 Hilbert 包络 |
| learn2learn 版本兼容性 | 保持当前依赖版本，不在本次变更中升级 |

---

# 11. 运行流程总图

```
┌─────────────────────────────────────────────────────────────┐
│                    离线预处理阶段 (一次性)                      │
│                                                             │
│  PU .mat → pu_loader → 滑窗 → 归一化 → 三通道构造             │
│     → sparse_coding → normalize255 → reshape(64,64)          │
│     → RGB fusion → 保存 PNG 到磁盘                            │
│                                                             │
│  输出: ./pu_data_processed/                                  │
│          ├── N09_M07_F10/    (source condition)              │
│          │     ├── fault_A/*.png                             │
│          │     ├── fault_B/*.png                             │
│          │     └── ...                                       │
│          ├── N15_M01_F10/    (target condition)              │
│          │     └── ...                                       │
│          └── ...                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    在线训练阶段                                │
│                                                             │
│  pu_dataset.py 加载 PNG → 归一化 /255 → [N,3,64,64]          │
│     → MAML_Dataset 兼容接口                                  │
│     → l2l.MetaDataset                                        │
│     → l2l.TaskDataset (FusedNWaysKShots)                     │
│     → MAML_learner.train() / test()                          │
│                                                             │
│  Inner loop:  clone → support set → adapt → query set → eval │
│  Outer loop:  accumulate grads → meta-update                 │
│                                                             │
│  输出: 模型 checkpoint + 跨工况诊断准确率                       │
└─────────────────────────────────────────────────────────────┘
```

---

# 文档版本

| 版本 | 日期 | 变更说明 |
|---|---|---|
| v1.0 | 2026-05-13 | 初始版本，结合可行性分析与源代码实现细节 |
