"""
帕德博恩轴承数据集（PU）上的 STFT-CNN4-MAML 基准模型配置文件。

当前基准流程：
原始 PU 振动信号 -> STFT 灰度时频图 -> CNN4Backbone 特征提取 -> MAML 少样本训练。

说明：
1. 这里的 32 类是整个故障类别池。
2. `n_way = 5` 表示每个 MAML episode 从 32 类中随机抽取 5 类。
3. `k_shot = 1` 表示每类 support 样本数为 1，因此当前任务是 5-way 1-shot。
"""

PU_CONFIG = {
    # ==================== 路径配置 ====================
    # 原始帕德博恩数据集路径，典型结构如：
    # K001/K001/N09_M07_F10_K001_1.mat
    'raw_data_root': r'C:\数据集\德国帕德博恩轴承数据集',

    # 代码内部使用的规范化 .mat 数据目录，整理后结构为：
    # pu_data_mat/工况名/故障类别.mat
    'mat_root': './pu_data_mat',

    # STFT 离线预处理后生成的图片目录，结构为：
    # pu_data_processed/工况名/故障类别/stft_000.png
    'root_path': './pu_data_processed',

    # ==================== 工况划分配置 ====================
    # 源域工况：用于 meta-train。这里使用多源工况训练，提高任务多样性。
    'source_condition': ['N15_M07_F04', 'N15_M07_F10', 'N15_M01_F10'],

    # 目标域工况：用于 validation/test。该工况内部再按类别划分验证集和测试集。
    'target_condition': 'N09_M07_F10',

    # 目标域每个类别中，多少比例样本用于 validation，剩余用于 test。
    'target_val_ratio': 0.5,

    # ==================== 原始数据整理配置 ====================
    # 每个故障类别、每个工况通常有多个 trial。基准模型先选一个 trial，保持实验轻量。
    # 'min' 表示选择编号最小的 trial；也可以改为 'max' 或具体编号字符串。
    'trial_pick': 'min',

    # 重新整理 pu_data_mat 时，如果目标 .mat 已存在，是否覆盖。
    'overwrite_mat_layout': True,

    # ==================== STFT 预处理配置 ====================
    # 预处理模式：一维振动信号切片后做 STFT，取 log 幅值谱并保存为灰度图。
    'preprocess_mode': 'stft_log',

    # 输入图片通道数。当前 STFT 图片为单通道灰度图。
    'in_channels': 1,

    # 每个样本从原始振动信号中截取的点数。
    'window_size': 4096,

    # STFT 图片缩放后的边长，最终输入为 [1, 64, 64]。
    'img_size': 64,

    # 每个“工况-故障类别”生成的 STFT 图片数量。
    # 当前 target 采用 50/50 划分，因此每类验证/测试各约 20 张。
    'samples_per_class': 40,

    # STFT 样本截取时的随机扰动种子。
    'preprocess_seed': 24,

    # 帕德博恩数据集中 vibration_1 通道对应的采样频率。
    'stft_fs': 64000,

    # STFT 每个短时窗的长度。
    'stft_nperseg': 256,

    # STFT 相邻短时窗之间的重叠长度。
    'stft_noverlap': 192,

    # STFT 使用的窗函数。
    'stft_window': 'hann',

    # 可选带通滤波范围，例如 (500, 12000)。None 表示当前不做带通滤波。
    'stft_bandpass': None,

    # log 幅值谱中的稳定项，避免 log(0)。
    'stft_log_eps': 1e-8,

    # 灰度图鲁棒归一化的低/高百分位，减少极端值影响。
    'stft_p_low': 5,
    'stft_p_high': 95,

    # 灰度 gamma 增强系数，小于 1 会提高较弱纹理的可见度。
    'stft_gamma': 0.7,

    # ==================== MAML 任务配置 ====================
    # 每个 episode 随机抽取的类别数，即 5-way。
    'n_way': 5,

    # 每类 support 样本数，即 1-shot。
    'k_shot': 1,

    # 每类 query 样本数。当前每个 episode 的 query 总数为 5 * 15 = 75。
    'q_query': 15,

    # MAML 内循环学习率，用于 support 集快速适应。
    'inner_lr': 0.05,

    # MAML 外循环学习率，用于 query 集元优化。
    'outer_lr': 0.005,

    # 正式训练轮数。
    'epochs': 1000,


    # 每轮可采样任务数量。
    'tasks_per_epoch': 1000,

    # 每次外循环累积多少个 episode 再更新一次参数。
    'meta_batch_size': 16,

    # 测试阶段 episode 数量。
    'test_meta_batch_size': 1000,

    # 测试阶段每个 episode 在 support 集上的内循环适应步数。
    'test_inner_steps': 10,

    # 不同 shot 设置下的训练阶段内循环步数。
    'adaptation_steps': {
        1: 3,
        5: 1,
    },

    # 全局随机种子。改动后会影响 episode 采样和训练随机性。
    'seed': 24,

    # ==================== 可视化输出配置 ====================
    # 训练曲线、测试混淆矩阵、t-SNE 和预测明细的保存目录。
    'visualization_dir': './results',

    # 是否在训练/测试后自动保存可视化结果。
    'save_visualizations': True,

    # t-SNE 最多使用的 query 特征点数，过大时会明显变慢。
    'tsne_max_points': 1000,

    # t-SNE perplexity。实际运行时会根据样本数自动修正到合法范围。
    'tsne_perplexity': 30,

    # t-SNE 只展示指定的 5 个原始故障类别，避免 32 类同时显示过于杂乱。
    # 如果想换展示类别，直接改这里的类别名即可，类别名必须与 pu_data_processed 下的文件夹名一致。
    'tsne_selected_classes': ['K001', 'KA01', 'KA07', 'KB23', 'KI16'],
}

# 快速冒烟测试开关。也可以通过运行脚本的 --quick 参数临时启用。
QUICK_TEST = False
