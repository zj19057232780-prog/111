"""
帕德博恩轴承数据集（PU）上的 STFT-CNN4-MAML 基准模型配置文件。

当前流程：
原始 PU 振动信号 -> STFT 灰度时频图 -> 可配置特征提取器（CNN4/LSK-lite）-> MAML 少样本训练。

说明：
1. 原始目录中的 32 个轴承编号会按分类预设合并为具有诊断语义的状态类别。
2. `n_way = 5` 表示每个 MAML episode 从当前状态类别池中随机抽取 5 类。
3. `k_shot = 1` 表示每类 support 样本数为 1，因此当前任务是 5-way 1-shot。
"""


PU_CLASSIFICATION_PRESETS = {
    # 实验 A：与 Semi-Meta-SGC 的 PU 人工损伤状态设置接近。
    'artificial_9state': {
        'description': '9-state artificial-damage diagnosis; random 5-way episodes',
        'class_groups': {
            'Healthy': ['K001'],
            'OR_EDM_L1': ['KA01'],
            'OR_Engraver_L1': ['KA05'],
            'OR_Engraver_L2': ['KA03'],
            'OR_Drilling_L1': ['KA07'],
            'OR_Drilling_L2': ['KA08'],
            'IR_EDM_L1': ['KI01'],
            'IR_Engraver_L1': ['KI03'],
            'IR_Engraver_L2': ['KI07'],
        },
        'tsne_selected_classes': [
            'Healthy',
            'OR_EDM_L1',
            'OR_Engraver_L2',
            'IR_EDM_L1',
            'IR_Engraver_L2',
        ],
    },
    # 实验 B：覆盖全部 32 个轴承，区分人工损伤、真实损伤和复合损伤。
    'full_6state': {
        'description': '6-state artificial/real/compound diagnosis; random 5-way episodes',
        'class_groups': {
            'Healthy': ['K001', 'K002', 'K003', 'K004', 'K005', 'K006'],
            'Artificial_OR': ['KA01', 'KA03', 'KA05', 'KA06', 'KA07', 'KA08', 'KA09'],
            'Real_OR': ['KA04', 'KA15', 'KA16', 'KA22', 'KA30'],
            'Artificial_IR': ['KI01', 'KI03', 'KI05', 'KI07', 'KI08'],
            'Real_IR': ['KI04', 'KI14', 'KI16', 'KI17', 'KI18', 'KI21'],
            'Compound': ['KB23', 'KB24', 'KB27'],
        },
        'tsne_selected_classes': [
            'Healthy',
            'Artificial_OR',
            'Real_OR',
            'Artificial_IR',
            'Real_IR',
        ],
    },
}


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
    'source_condition': ['N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10'],

    # 目标域工况：用于 validation/test。该工况内部再按类别划分验证集和测试集。
    'target_condition': 'N09_M07_F10',

    # 目标域每个类别中，多少比例样本用于 validation，剩余用于 test。
    'target_val_ratio': 0.5,

    # ==================== 分类任务切换 ====================
    # 只修改这一项即可切换分类实验：
    # - 'artificial_9state'：9 种人工损伤状态类别池，每个 episode 随机抽 5 类。
    # - 'full_6state'：人工/真实/复合损伤 6 类类别池，每个 episode 随机抽 5 类。
    'classification_preset': 'artificial_9state',

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

    # ==================== 模型骨干配置 ====================
    # backbone 可选：
    # - 'cnn4'：原始基准模型，4 层传统 CNN。
    # - 'lsk_lite'：模块 1，轻量化大核选择卷积网络，用于替换 CNN4。
    'backbone': 'lsk_lite',

    # 默认模型保存名前缀。实际文件名会自动追加分类预设名，避免实验互相覆盖。
    'model_name_base': 'STFT_LSKLite_GFNetLite_ARSM_MAML',

    # LSK-lite 三个阶段的通道数。最后一个数也是 GAP 后的特征维度。
    'lsk_stage_channels': (32, 64, 96),

    # LSK-lite 每个阶段堆叠的 LSK block 数。先用轻量设置，适合 MAML。
    'lsk_stage_depths': (1, 1, 1),

    # LSK block 内 MLP 扩展倍率，越大参数越多。
    'lsk_mlp_ratio': 2,

    # ==================== 频域增强模块配置 ====================
    # frequency_module 可选：
    # - 'none'：不启用频域增强，得到 STFT-LSKLite-MAML，用于消融对照。
    # - 'gfnet_lite'：启用轻量 GFNet 全局频域滤波增强模块。
    'frequency_module': 'none',

    # GFNetLite 堆叠层数。先用 1 层，避免 MAML 训练过慢或过拟合。
    'gfnet_depth': 1,

    # GFNetLite 内部 MLP 扩展倍率。
    'gfnet_mlp_ratio': 2,

    # 可学习复数频域滤波器的初始化尺度，参考 GFNet 使用较小初始化。
    'gfnet_weight_scale': 0.02,

    # GFNetLite 残差分支缩放初值，较小值有助于稳定接入已跑通的 LSK 特征。
    'gfnet_layer_scale_init': 1e-2,

    # ==================== 自适应去噪模块配置 ====================
    # denoise_module 可选：
    # - 'none'：关闭去噪模块，恢复 STFT-LSKLite-GFNetLite-MAML。
    # - 'arsm'：启用自适应残差收缩模块，学习逐通道软阈值。
    'denoise_module': 'none',

    # ARSM 阈值生成网络的通道压缩比例。96 通道下 hidden=24。
    'arsm_reduction': 4,

    # ARSM 从恒等映射向软阈值输出插值的初始比例。
    # 小值初始化可避免在二阶 MAML 训练初期过度改变已有特征。
    'arsm_blend_init': 1e-3,

    # ==================== 注意力模块配置 ====================
    # attention_module 可选：
    # - 'none'：不启用注意力模块，用于 LSK/GFNet 前序消融。
    # - 'ema_lite'：启用轻量 EMA 跨空间多尺度注意力模块。
    # - 'gcnet'：启用 GCNet 全局上下文模块，用于替代 EMA 做第三模块消融。
    'attention_module': 'none',

    # EMA-lite 分组数。96 通道下默认 8 组，每组 12 通道，兼顾轻量和表达能力。
    'ema_factor': 8,

    # EMA-lite 残差分支缩放初值。先用较小值，降低对 GFNet/LSK 特征的初始扰动。
    'ema_layer_scale_init': 1e-3,

    # GCNet 通道压缩比例。96 通道下 0.25 对应 24 个上下文隐藏通道，参数量较小。
    'gcnet_ratio': 0.125,

    # GCNet 上下文池化方式：'att' 为注意力加权全局池化，'avg' 为平均池化。
    'gcnet_pooling_type': 'att',

    # GCNet 融合方式。默认只用 channel_add，初始近似恒等映射，比 channel_mul 更稳。
    'gcnet_fusion_types': ('channel_mul',),

    # GCNet 残差分支缩放初值。用于限制全局上下文分支后期过强，降低 loss 爆炸和 NaN 风险。
    'gcnet_layer_scale_init': 1e-4,

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

    # 外循环采用余弦退火，避免模型接近收敛后仍以固定高学习率更新而发散。
    'outer_lr_scheduler': 'cosine',

    # 余弦退火结束时的最小外循环学习率。
    'outer_lr_min': 5e-5,

    # 二阶 MAML 在 GFNet 频域分支上可能产生梯度尖峰，更新前执行全局范数裁剪。
    # None 或小于等于 0 表示关闭裁剪。
    'outer_grad_clip_norm': 1.0,

    # 正式训练轮数。
    'epochs': 500,


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
    'seed':38,



    # ==================== 可视化输出配置 ====================
    # 训练曲线、测试混淆矩阵、t-SNE 和预测明细的保存目录。
    'visualization_dir': './results',

    # 是否在训练/测试后自动保存可视化结果。
    'save_visualizations': True,

    # t-SNE 最多使用的 query 特征点数，过大时会明显变慢。
    'tsne_max_points': 1000,

    # t-SNE perplexity。实际运行时会根据样本数自动修正到合法范围。
    'tsne_perplexity': 30,

    # None 表示自动使用当前分类预设定义的 5 个语义类别。
    # 也可以手动填写当前预设中的类别名列表。
    'tsne_selected_classes': None,
}


def resolve_pu_config(config):
    """Expand the selected classification preset into a runtime config."""
    resolved = dict(config)
    preset_name = resolved.get('classification_preset', 'artificial_9state')
    if preset_name not in PU_CLASSIFICATION_PRESETS:
        available = ', '.join(PU_CLASSIFICATION_PRESETS)
        raise ValueError(
            f'Unknown classification_preset: {preset_name}. '
            f'Available presets: {available}'
        )

    preset = PU_CLASSIFICATION_PRESETS[preset_name]
    resolved['class_groups'] = {
        class_name: list(bearing_ids)
        for class_name, bearing_ids in preset['class_groups'].items()
    }
    resolved['classification_description'] = preset['description']
    previous_preset = resolved.get('_resolved_classification_preset')
    if (
        not resolved.get('tsne_selected_classes')
        or (previous_preset is not None and previous_preset != preset_name)
    ):
        resolved['tsne_selected_classes'] = list(preset['tsne_selected_classes'])
    resolved['_resolved_classification_preset'] = preset_name

    model_name_base = resolved.get(
        'model_name_base',
        resolved.get('model_name', 'STFT_CNN4_MAML'),
    )
    resolved['model_name'] = f'{model_name_base}_{preset_name}'
    return resolved


PU_CONFIG = resolve_pu_config(PU_CONFIG)

# 快速冒烟测试开关。也可以通过运行脚本的 --quick 参数临时启用。
QUICK_TEST = False
