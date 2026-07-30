"""CWRU 跨工况少样本故障诊断的独立配置。

本文件只服务于 ``run_cwru_maml.py``，不会修改 PU 实验配置。
当前默认模块组合为 E12：LSK-lite + ARSM，seed=24，10-way 1-shot。

服务器运行示例：
    python -u run_cwru_maml.py --train --test \
        --model_path ./model_save/CWRU-E12-seed24

切换实验时，必须同时核对模块开关、seed 和 ``--model_path`` 中的 Run ID，
避免模型实际结构与 E 编号不一致。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


# ==================== 十状态分类定义 ====================
# Healthy：正常轴承。
# IR / Ball / OR：分别表示内圈、滚动体和外圈故障。
# 007 / 014 / 021：故障直径分别为 0.007、0.014、0.021 英寸。
# 外圈故障统一使用 6 点钟方向数据（OuterRace6）。
# 当前流程只读取 48 kHz 驱动端振动信号，原始 MAT 变量应以 DE_time 结尾。
CWRU_CLASS_SPECS = {
    'Healthy': {
        'subset': 'NormalBaseline',
        'filename': 'Normal.mat',
    },
    'IR_007': {
        'subset': '48DriveEndFault',
        'filename': '0.007-InnerRace.mat',
    },
    'IR_014': {
        'subset': '48DriveEndFault',
        'filename': '0.014-InnerRace.mat',
    },
    'IR_021': {
        'subset': '48DriveEndFault',
        'filename': '0.021-InnerRace.mat',
    },
    'Ball_007': {
        'subset': '48DriveEndFault',
        'filename': '0.007-Ball.mat',
    },
    'Ball_014': {
        'subset': '48DriveEndFault',
        'filename': '0.014-Ball.mat',
    },
    'Ball_021': {
        'subset': '48DriveEndFault',
        'filename': '0.021-Ball.mat',
    },
    'OR_007': {
        'subset': '48DriveEndFault',
        'filename': '0.007-OuterRace6.mat',
    },
    'OR_014': {
        'subset': '48DriveEndFault',
        'filename': '0.014-OuterRace6.mat',
    },
    'OR_021': {
        'subset': '48DriveEndFault',
        'filename': '0.021-OuterRace6.mat',
    },
}


# ==================== 工况与转速目录映射 ====================
# C0/C1/C2/C3 分别对应 0/1/2/3 hp 电机负载；负载升高时转速逐步降低。
# 字典值是原始数据集中的转速文件夹名，用于定位 MAT 文件。
CWRU_CONDITIONS = {
    'C0_1797rpm': '1797',
    'C1_1772rpm': '1772',
    'C2_1750rpm': '1750',
    'C3_1730rpm': '1730',
}


CWRU_CONFIG = {
    # ==================== 数据路径 ====================
    # 原始数据目录应位于当前项目的 CWRU/ 下，典型结构为：
    # CWRU/48DriveEndFault/1797/0.007-InnerRace.mat
    'raw_data_root': str(PROJECT_ROOT / 'CWRU'),

    # STFT 离线预处理输出目录。训练直接读取这里的 64×64 灰度 PNG。
    # 使用 --preprocess --clean 会先删除此目录再完整重建，请勿在其中放其他文件。
    'root_path': str(PROJECT_ROOT / 'cwru_data_processed'),

    # ==================== 跨工况协议 ====================
    # 三个源工况用于 meta-train；C3 完整留作目标工况。
    # 目标工况原始信号前半段生成 validation，后半段生成 test，避免二者窗口重叠。
    # 注意：当前训练脚本仍会用目标 validation 准确率选择最佳 checkpoint；若论文声称
    # 严格 target-unseen domain generalization，应改成源域验证，目标域只做最终测试。
    'condition_speeds': dict(CWRU_CONDITIONS),
    'source_condition': ['C0_1797rpm', 'C1_1772rpm', 'C2_1750rpm'],
    'target_condition': 'C3_1730rpm',

    # 协议标识及类别表。通常无需手动修改。
    'classification_preset': 'cwru_10state_3source_1target',
    'class_specs': dict(CWRU_CLASS_SPECS),
    'class_names': list(CWRU_CLASS_SPECS),

    # ==================== STFT 预处理 ====================
    # 48 kHz 驱动端信号；与 PU 保持相近物理时间尺度：
    # 每张图覆盖 3072/48000=64 ms；STFT 窗长 192 点=4 ms；
    # hop=192-144=48 点=1 ms。
    'preprocess_mode': 'stft_log',
    'signal_channel': 'DE_time',
    'stft_fs': 48000,
    'window_size': 3072,
    'stft_nperseg': 192,
    'stft_noverlap': 144,
    'stft_window': 'hann',

    # None 表示不额外截取频带，保留 STFT 全频段。
    'stft_bandpass': None,

    # 对幅值取对数时的数值稳定项。
    'stft_log_eps': 1e-8,

    # 用第5/95百分位做灰度拉伸，降低极端幅值对图像对比度的影响。
    'stft_p_low': 5,
    'stft_p_high': 95,

    # gamma<1 会增强较弱时频纹理。
    'stft_gamma': 0.7,

    # 每个信号窗口先减均值，再计算 STFT。
    'normalize': 'demean',
    'img_size': 64,
    'in_channels': 1,

    # 每个源工况、每个类别生成40张；目标工况每类生成20张验证图和20张测试图。
    # 当前10类总图片数：3×10×40 + 10×(20+20) = 1600。
    'source_samples_per_class': 40,
    'target_val_samples_per_class': 20,
    'target_test_samples_per_class': 20,

    # 只控制预处理窗口位置的随机采样；正式模块对比期间应固定不变。
    'preprocess_seed': 24,

    # ==================== 原始数据异常修正 ====================
    # 下载包的 metadata.txt 将该文件指向 X174，但实际 MAT 中对应变量为 X173_DE_time。
    # 这是已核实的数据例外，不是可调超参数。
    'variable_overrides': {
        '48DriveEndFault/1797/0.014-InnerRace.mat': 'X173_DE_time',
    },

    # ==================== 模型与模块消融 ====================
    # CWRU 模型配置与 config_pu.py 完全独立。
    # 九个实验编号对应关系：
    # E00: CNN4
    # E10: LSK
    # E11: LSK + GFNet
    # E12: LSK + ARSM（当前默认）
    # E13: LSK + EMA
    # E14: LSK + GCNet
    # E21: LSK + GFNet + ARSM
    # E22: LSK + GFNet + EMA
    # E23: LSK + GFNet + GCNet

    # backbone 可选：'cnn4' / 'lsk_lite'。
    'backbone': 'lsk_lite',

    # 未传 --model_path 时使用的默认文件名前缀。正式实验建议始终显式传入
    # ./model_save/CWRU-E<编号>-seed<种子>，脚本会自动追加 _best。
    'model_name': 'CWRU_STFT_LSKLite_ARSM_MAML_10state_C012_to_C3',

    # CNN4 每层通道数；LSK-lite 不使用此值作为阶段通道配置。
    'hidden_size': 64,

    # LSK-lite 三阶段通道、每阶段 block 数和 MLP 扩展倍率。
    'lsk_stage_channels': (32, 64, 96),
    'lsk_stage_depths': (1, 1, 1),
    'lsk_mlp_ratio': 2,

    # 频域增强可选：'none' / 'gfnet_lite'。
    'frequency_module': 'none',
    'gfnet_depth': 1,
    'gfnet_mlp_ratio': 2,
    'gfnet_weight_scale': 0.02,
    'gfnet_layer_scale_init': 1e-2,

    # 去噪模块可选：'none' / 'arsm'。
    'denoise_module': 'arsm',
    'arsm_reduction': 4,
    'arsm_blend_init': 1e-3,

    # 注意力模块可选：'none' / 'ema_lite' / 'gcnet'。
    # EMA 与 GCNet 是互斥选项，不能在本字段中同时启用。
    'attention_module': 'none',
    'ema_factor': 8,
    'ema_layer_scale_init': 1e-3,

    # GCNet 使用注意力全局池化、乘性上下文融合和近恒等的小尺度残差注入。
    'gcnet_ratio': 0.125,
    'gcnet_pooling_type': 'att',
    'gcnet_fusion_types': ('channel_mul',),
    'gcnet_layer_scale_init': 1e-4,

    # ==================== MAML episode 与优化 ====================
    # 每个 episode 使用全部10类，避免5-way随机类别组合造成任务难度波动。
    'n_way': 10,

    # 每类 support 样本数。当前为 10-way 1-shot。
    'k_shot': 1,

    # 每类 query 样本数，所以每个 episode 有 10×15=150 个 query。
    'q_query': 15,

    # inner_lr：support 内循环快速适应学习率；outer_lr：query 外循环学习率。
    'inner_lr': 0.05,
    'outer_lr': 0.005,

    # 外循环使用余弦退火，从 outer_lr 逐步降到 outer_lr_min。
    'outer_lr_scheduler': 'cosine',
    'outer_lr_min': 5e-5,

    # 外循环梯度范数裁剪上限，用于抑制二阶MAML偶发梯度尖峰。
    'outer_grad_clip_norm': 1.0,
    'epochs': 500,

    # 兼容 TaskDataset 的任务数配置。注意：当前训练循环每个 epoch 实际只调用
    # meta_batch_size 次 sample()；此值目前不会让每个 epoch 真正执行1000个episode。
    'tasks_per_epoch': 1000,

    # 每次外循环更新累计的 episode 数；当前每个 epoch 实际处理16个训练episode。
    'meta_batch_size': 16,

    # 正式测试的随机 episode 数。
    'test_meta_batch_size': 1000,

    # 测试时每个 episode 在 support 上执行10步适应。
    'test_inner_steps': 10,

    # 训练时不同 shot 设置对应的内循环步数：1-shot用3步，5-shot用1步。
    'adaptation_steps': {
        1: 3,
        5: 1,
    },

    # 控制模型初始化和训练/episode随机采样。三seed正式实验依次改为3、24、38。
    # 该值不改变预处理窗口，预处理由 preprocess_seed 单独控制。
    'seed': 24,

    # ==================== 结果可视化 ====================
    # 每个 Run 会按 --model_path 的文件名在 results/cwru/ 下建立独立子目录。
    'visualization_dir': str(PROJECT_ROOT / 'results' / 'cwru'),
    'save_visualizations': True,

    # t-SNE最多使用1000个点；perplexity必须小于实际样本数。
    'tsne_max_points': 1000,
    'tsne_perplexity': 30,

    # 固定展示的代表类别，不影响训练和总体测试准确率。
    'tsne_selected_classes': [
        'Healthy',
        'IR_007',
        'IR_021',
        'Ball_014',
        'OR_021',
    ],
}


# 全局快速模式开关。通常保持False，服务器冒烟测试请显式传入 --quick；
# quick模式只训练3个epoch、测试4个episode，不得记录为正式结果。
QUICK_TEST = False
