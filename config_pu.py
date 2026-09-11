"""
帕德博恩轴承数据集（PU）的 STFT-MAML 统一配置文件。

当前流程：
原始 PU 振动信号
-> STFT 灰度时频图
-> 可配置特征提取器（CNN4 / LSK-lite + 可选增强模块）
-> 标准等权 MAML 或 GGM-RW 加权 MAML。

说明：
1. 原始目录中的 32 个轴承编号会按分类预设合并为具有诊断语义的状态类别。
2. `n_way = 5` 表示每个 MAML episode 从当前状态类别池中随机抽取 5 类。
3. `k_shot = 1` 表示每类 support 样本数为 1，因此当前任务是 5-way 1-shot。
4. 当前默认实验为：9 类状态池 + LSK-lite + GCNet + GGM-RW。
5. 目标工况的 validation 标签用于逐 epoch 选取 best checkpoint；test 不参与训练或选模。
"""


PU_CLASSIFICATION_PRESETS = {
    # 实验 A：9 类人工损伤状态池。
    # 每个语义状态只选择一个具有代表性的轴承编号，因此实际使用 9 个轴承编号，
    # 并不是把 PU 原始目录中的 32 个轴承编号全部当作独立类别。
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
    # 实验 B：把全部 32 个轴承编号合并成 6 个语义状态类别。
    # 同一语义类中的多个轴承编号会在数据加载时合并，不会被当作不同类别。
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
    # ==================== 经典模型统一接口 ====================
    # configured_maml：保留下方当前主干、增强模块和GGM-RW配置（旧模型兼容）。
    # maml / protonet / matchingnet / relationnet：经典CNN4基线，自动关闭增强和任务加权。
    # 只改method即可切换算法；每个算法必须先训练自己的权重。
    'method': 'resnet18',
    # 无命令行动作参数时执行此模式：train / test / train_test。
    # 显式--train、--test仍优先；test只加载当前model_name对应的_best权重。
    'run_mode': 'train_test',
    # 普通ResNet18：源域从零监督训练，目标全类别直接测试，无support微调。
    'supervised_epochs': 100,
    'supervised_batch_size': 64,
    'supervised_lr': 0.001,
    'supervised_weight_decay': 1e-4,
    'supervised_max_batches': 0,  # 0=全量；正数只用于调试

    # 非元学习：method='resnet18_ft'（STFT）或'tl_wdcnn'（原始一维振动）。
    # 源域普通监督预训练，目标每个episode独立重置模型和局部分类头。
    'ft_pretrain_epochs': 100,
    'ft_pretrain_batch_size': 64,
    'ft_pretrain_lr': 0.001,
    'ft_weight_decay': 1e-4,
    'ft_pretrain_max_batches': 0,  # 0=每轮遍历全部源样本；正数仅用于限量调试
    'ft_validation_episodes': 16,
    'ft_validation_seed': 2026,    # 每轮固定验证任务与局部头初始化
    'ft_patience': 0,             # 0=不提前停止；正数为验证无提升轮数
    'ft_steps': 10,               # support微调步数，与MAML inner_steps独立
    'ft_lr': 0.01,
    'ft_scope': 'all',            # all=全网微调；head=冻结编码器仅训练新头
    'ft_bn_mode': 'batch',        # batch=批统计，与现有query批量评测一致；frozen=源域统计
    'metric_temperature': 1.0,  # ProtoNet距离/MatchingNet余弦注意力温度
    'matching_fce': True,       # MatchingNet完整上下文嵌入；False为简化版
    'matching_fce_steps': 5,    # 查询样本的注意力LSTM读取步数，不是梯度适配
    'relation_hidden_size': 8,  # RelationNet关系头全连接隐藏维度

    # ==================== 路径配置 ====================
    # 原始帕德博恩数据集路径，只在重新整理 .mat 数据时使用。典型结构如：
    # K001/K001/N09_M07_F10_K001_1.mat
    'raw_data_root': r'C:\数据集\德国帕德博恩轴承数据集',

    # 代码内部使用的规范化 .mat 数据目录，只在STFT预处理阶段使用。整理后结构为：
    # pu_data_mat/工况名/故障类别.mat
    'mat_root': './pu_data_mat',

    # 训练和测试直接读取的STFT图片目录，结构为：
    # pu_data_processed/工况名/故障类别/stft_000.png
    'root_path': './pu_data_processed',

    # ==================== 工况划分配置 ====================
    # 源域工况：仅用于meta-train。三个工况的同类样本会合并后构造训练episode。
    'source_condition': ['N15_M01_F10', 'N15_M07_F04', 'N15_M07_F10'],

    # 目标域工况：不参与梯度训练，但内部数据会拆分为validation和test。
    # validation用于训练过程监控和best checkpoint选择；test只在最终测试时使用。
    'target_condition': 'N09_M07_F10',

    # 目标域每个轴承编号中用于validation的比例，剩余样本用于test。
    # 当前每个轴承编号有40张图，因此9类预设下每类约为20张validation+20张test。
    'target_val_ratio': 0.5,

    # ==================== PU 分类任务切换 ====================
    # 只修改这一项即可切换类别池；修改后旧checkpoint不再兼容，必须重新训练：
    # - 'artificial_9state'：9 种人工损伤状态类别池，每个 episode 随机抽 5 类。
    # - 'full_6state'：人工/真实/复合损伤 6 类类别池，每个 episode 随机抽 5 类。
    'classification_preset': 'artificial_9state',

    # ==================== 原始数据整理配置 ====================
    # 每个轴承编号、每个工况通常有多个trial。当前只选一个trial，保持实验规模一致。
    # 'min' 表示选择编号最小的 trial；也可以改为 'max' 或具体编号字符串。
    'trial_pick': 'min',

    # 重新整理 pu_data_mat 时，如果目标 .mat 已存在，是否覆盖。
    'overwrite_mat_layout': True,

    # ==================== STFT 预处理配置 ====================
    # 预处理模式：一维振动信号切片后做 STFT，取 log 幅值谱并保存为灰度图。
    'preprocess_mode': 'stft_log',

    # 输入图片通道数。当前STFT图片为单通道灰度图，不应改成3，除非重新设计模型输入。
    'in_channels': 1,

    # ==================== 模型骨干配置 ====================
    # backbone决定主特征提取器，可选：
    # - 'cnn4'：原始基准模型，4 层传统 CNN。
    # - 'lsk_lite'：模块 1，轻量化大核选择卷积网络，用于替换 CNN4。
    'backbone': 'lsk_lite',

    # 默认模型保存名前缀；运行时自动追加分类预设名。
    # 命令行传入--model_path时，以命令行路径为准。
    'model_name_base': 'STFT_LSKLite_GCNet_GGMRW_MAML',

    # LSK-lite 三个阶段的通道数。最后一个数也是 GAP 后的特征维度。
    'lsk_stage_channels': (32, 64, 96),

    # LSK-lite 每个阶段堆叠的 LSK block 数。先用轻量设置，适合 MAML。
    'lsk_stage_depths': (1, 1, 1),

    # LSK block 内 MLP 扩展倍率，越大参数越多。
    'lsk_mlp_ratio': 2,

    # ==================== 频域增强模块配置 ====================
    # frequency_module位于LSK主干之后、去噪/注意力模块之前，可选：
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
    # denoise_module位于频域模块之后、注意力模块之前，可选：
    # - 'none'：关闭去噪模块，恢复 STFT-LSKLite-GFNetLite-MAML。
    # - 'arsm'：启用自适应残差收缩模块，学习逐通道软阈值。
    'denoise_module': 'none',

    # ARSM 阈值生成网络的通道压缩比例。96 通道下 hidden=24。
    'arsm_reduction': 4,

    # ARSM 从恒等映射向软阈值输出插值的初始比例。
    # 小值初始化可避免在二阶 MAML 训练初期过度改变已有特征。
    'arsm_blend_init': 1e-3,

    # ==================== 注意力模块配置 ====================
    # attention_module位于特征提取末端、全局平均池化之前，可选：
    # - 'none'：不启用注意力模块，用于 LSK/GFNet 前序消融。
    # - 'ema_lite'：启用轻量 EMA 跨空间多尺度注意力模块。
    # - 'gcnet'：启用 GCNet 全局上下文模块，用于替代 EMA 做第三模块消融。
    'attention_module': 'gcnet',

    # EMA-lite 分组数。96 通道下默认 8 组，每组 12 通道，兼顾轻量和表达能力。
    'ema_factor': 8,

    # EMA-lite 残差分支缩放初值。先用较小值，降低对 GFNet/LSK 特征的初始扰动。
    'ema_layer_scale_init': 1e-3,

    # GCNet通道压缩比例。当前末级96通道，0.125对应12个上下文隐藏通道。
    'gcnet_ratio': 0.125,

    # GCNet 上下文池化方式：'att' 为注意力加权全局池化，'avg' 为平均池化。
    'gcnet_pooling_type': 'att',

    # GCNet融合方式。当前仅使用channel_mul；模块末层零初始化并配合很小的layer scale，
    # 因此初始状态仍近似恒等映射。
    'gcnet_fusion_types': ('channel_mul',),

    # GCNet 残差分支缩放初值。用于限制全局上下文分支后期过强，降低 loss 爆炸和 NaN 风险。
    'gcnet_layer_scale_init': 1e-4,

    # ==================== GGM-RW 任务重加权 ====================
    # 'none'：标准等权 MAML。
    # 'ggm_rw'：只在meta-train outer loop中启用Generalization-Gap Guided
    # Meta Reweighting。它改变训练任务query loss的聚合权重，但不改变episode的
    # support/query数量、正式二阶MAML内循环、validation或test。
    'task_weighting_mode': 'ggm_rw',

    # meta-batch内任务难度softmax的温度。越小权重差异越明显，越大越接近均匀。
    'task_weight_temperature': 1.0,

    # 困难度分布与均匀权重的最大混合比例。
    # 0表示完全等权，1表示完全使用gap产生的softmax分布；当前0.20属于温和重加权。
    'task_weight_alpha_max': 0.20,

    # alpha通过cosine warm-up从0平滑增长到alpha_max所需的epoch数。
    # 该warm-up只控制任务权重强度，不是下面的outer learning-rate scheduler。
    'task_weight_warmup_epochs': 50,

    # meta-batch内标准化后的generalization-gap z-score截断范围，用于抑制异常任务。
    'task_weight_z_clip': 2.0,

    # 单任务最大权重为task_weight_max_ratio/meta_batch_size。
    # 当前meta_batch_size=16时上限为3/16=0.1875，均匀权重为1/16=0.0625。
    'task_weight_max_ratio': 3.0,

    # gap标准化和权重归一化的数值稳定项，避免除零。
    'task_weight_eps': 1e-6,

    # 每个样本从原始振动信号中截取的点数。
    'window_size': 4096,

    # STFT 图片缩放后的边长，最终输入为 [1, 64, 64]。
    'img_size': 64,

    # 每个“工况-轴承编号”生成的STFT图片数量。
    # artificial_9state中每个状态对应一个轴承编号，因此目标域每类共40张。
    # full_6state会合并多个轴承编号，所以不同语义类的总样本数并不相同。
    'samples_per_class': 40,

    # STFT预处理阶段的随机种子；只影响重新生成图片，不影响训练episode采样。
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
    # 每个episode从当前类别池随机抽取的类别数，即5-way。
    'n_way': 5,

    # 每类 support 样本数，即 1-shot。
    'k_shot': 1,

    # 每类 query 样本数。当前每个 episode 的 query 总数为 5 * 15 = 75。
    'q_query': 15,

    # MAML内循环学习率：每个episode在support集上进行参数快速适应。
    'inner_lr': 0.05,

    # MAML外循环初始学习率：根据训练episode的query loss更新基础模型。
    'outer_lr': 0.005,

    # 外循环学习率调度。cosine会在全部epochs内从outer_lr平滑下降到outer_lr_min。
    'outer_lr_scheduler': 'cosine',

    # 余弦退火结束时的最小外循环学习率。
    'outer_lr_min': 5e-5,

    # 二阶MAML在复杂分支上可能产生梯度尖峰，opt.step()前执行全局范数裁剪。
    # None 或小于等于 0 表示关闭裁剪。
    'outer_grad_clip_norm': 1.0,

    # 正式训练轮数。
    'epochs': 500,


    # TaskDataset声明的可采样任务数量上限。
    # 注意：当前训练循环每个epoch实际只采样meta_batch_size个训练episode，
    # 并执行一次outer update；它不会在每个epoch遍历这里的1000个任务。
    'tasks_per_epoch': 1000,

    # 每次outer update聚合的训练episode数量；当前500轮共执行500次outer update，
    # 合计采样500*16=8000个训练episode。
    'meta_batch_size': 16,

    # ==================== 满分验证提前停止 ====================
    # True：常规validation达到下面的触发准确率后，追加一次更大规模的确认验证；
    # 确认通过就提前结束训练，随后主程序仍会加载_best执行最终test。
    # False：始终训练满epochs，不执行确认验证。
    'early_stop_on_perfect_validation': True,

    # 常规validation的触发阈值。1.0表示当轮16个episode全部预测正确时才触发。
    'early_stop_trigger_accuracy': 1.0,

    # 触发后追加采样的validation episode数。确认阶段仅评估，不更新模型；
    # RNG会在确认后恢复，因此确认失败不会改变后续训练episode序列。
    'early_stop_confirmation_episodes': 100,

    # 追加确认验证达到该准确率才真正停止。当前99%用于排除单轮容易任务造成的偶然满分。
    'early_stop_confirmation_accuracy': 0.99,

    # 最终test随机采样的episode数量，用于计算平均测试准确率。
    'test_meta_batch_size': 1000,

    # test阶段每个episode在support集上的内循环适应步数。
    # 当前为10步，高于训练/validation的3步；比较实验时必须保持该值一致。
    'test_inner_steps': 10,

    # 不同shot设置下，训练和validation使用相同的3次内循环适应，
    # 确保1/3/5-shot实验只改变support样本数，不同时改变适应深度。
    'adaptation_steps': {
        1: 3,
        3: 3,
        5: 3,
    },

    # 训练全局随机种子，同时影响目标validation/test拆分、episode采样、
    # 参数初始化和优化随机性；修改seed后应使用新的模型及结果名称。
    'seed': 38,



    # ==================== 可视化输出配置 ====================
    # 训练曲线、混淆矩阵、t-SNE和预测明细的根目录。
    # 实际输出会再按--model_path的末级名称建立独立子目录。
    'visualization_dir': './results',

    # 是否在训练/测试后自动保存可视化结果。
    'save_visualizations': True,

    # t-SNE 最多使用的 query 特征点数，过大时会明显变慢。
    'tsne_max_points': 1000,

    # t-SNE perplexity。实际运行时会根据样本数自动修正到合法范围。
    'tsne_perplexity': 30,

    # None表示自动使用当前分类预设预先指定的5个语义类别绘制固定类别t-SNE。
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
    from classic_config import resolve_method_config
    return resolve_method_config(resolved)


PU_CONFIG = resolve_pu_config(PU_CONFIG)

# 快速冒烟测试开关。也可以通过运行脚本的 --quick 参数临时启用。
QUICK_TEST = False
