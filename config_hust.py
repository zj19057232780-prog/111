"""
河内理工大学 HUST bearing v3 跨负载少样本实验配置。

数据流程：原始 MAT 的稳态 data → 非重叠窗口 → 64×64 灰度 STFT → MAML。
类别定义：6205–6208 的同一故障合并为一类，组成七类候选池；每次随机抽五类。
评测协议：两个源负载元训练，目标负载分成 validation 和 test；前者选模，后者最终测试。

使用提示：
1. 当前命令行入口会用 argparse 默认值覆盖 task/method/k_shot/seed，
   即使没有显式传参也是如此。通过 --task/--method/--shots/--seed 切换这些设置；
   仅修改下面这四项不会改变当前命令行入口的默认实验。
2. --quick 临时覆盖 epochs=1、meta_batch_size=1、validation_episodes=1、
   test_meta_batch_size=2；随后若传入 --test-episodes，则以该参数为准。
3. 预处理参数进入数据签名。改窗口、STFT、样本数等后，应使用新的
   --processed-root 并执行 --preprocess，旧数据不允许静默混用。
4. 正式默认：5-way 1-shot、每类15 query、500次外循环更新、1000个测试episode。
"""
from copy import deepcopy
from pathlib import Path

# 路径以本配置文件所在目录为基准，不依赖启动命令时的工作目录。
BASE = Path(__file__).resolve().parent

# 每项为（源负载列表，目标负载），单位 W；型号在两侧均为6205–6208。
TASKS = {
    'HUST-T1': ([0, 200], 400),
    'HUST-T2': ([0, 400], 200),
    'HUST-T3': ([200, 400], 0),
}

HUST_CONFIG = {
    # ==================== 数据路径与实验选择 ====================
    # 原始包根目录：递归查找 MAT，可包含下一层 HUST bearing dataset 文件夹。
    # 命令行 --raw-root 可覆盖；仅读取 data，不将启动信号 ru_raw 混入实验。
    'raw_data_root': str(BASE / 'HUST bearing a practical dataset for ball bearing fault diagnosis'),
    # 预处理总目录：实际图片/manifest位于其下的 HUST-T1、T2、T3 子目录。
    'processed_root': str(BASE / 'hust_data_processed'),
    # 任务默认值；命令行运行时用 --task 选择，--task all 顺序运行三个任务。
    'task': 'HUST-T1',
    # configured_maml：采用下方LSK-lite+GCNet+GGM-RW配置。
    # maml：标准CNN4-MAML；配置解析器自动关闭增强模块及GGM任务加权。
    # 当前HUST入口仅支持这两条主线；命令行用 --method 切换。
    'method': 'maml',
    # 控制模型初始化和源训练episode采样，计划使用3、24、38。
    # 不改变原始窗口划分或固定验证/测试清单；命令行用 --seed 切换。
    'seed': 24,

    # ==================== 全局类别池与每次episode ====================
    # 顺序对应全局标签0–6：正常、内圈、外圈、滚动体、内外圈、内圈+滚动体、外圈+滚动体。
    # IO/IB/OB各自是独立的复合故障类；不可仅按文件名首字母合并。
    'class_names': ['N', 'I', 'O', 'B', 'IO', 'IB', 'OB'],
    # 排除缺B/IB的6204；下面四种型号按同一故障状态合并，不作为四个分类标签。
    'bearing_types': [6205, 6206, 6207, 6208],
    # 分类预设名用于记录实验身份，不会调用PU的分类预设解析器。
    'classification_preset': 'hust_7state_4bearing',
    # 协议版本标识；与预处理、checkpoint元信息一起校验数据/实验是否匹配。
    'protocol': 'HUST-v3-C7-5way-LOLO-targetval-v1',
    # 七类中每次随机抽五类，分类头输出五维，episode内重新映射成0–4。
    # 当前解析器固定支持5-way，不通过改此数值切换7-way。
    'n_way': 5,
    # 每类support数量，仅支持1或5；命令行用 --shots 切换。
    'k_shot': 1,
    # 每类query数量，当前协议固定15；每episode共75个query。
    # 1-shot共5个support，5-shot共25个support；这不是整个目标数据池的大小。
    'q_query': 15,

    # ==================== 每条原始记录的窗口预算 ====================
    # 以下数量均按“型号×故障状态×负载”的单条MAT计数，不是合并后的每类总数。
    # 两个源负载：2×4型号×7类×40=2240张，合并后每类320张。
    'source_samples': 40,
    # 目标前半段用于validation：1×4×7×20=560张，每类80张，用于选模/早停。
    'validation_samples': 20,
    # 目标后半段用于test：同为560张、每类80张；与validation留一个窗口的隔离带。
    # validation和test各自独立抽取support/query，不能跨池取样。
    'test_samples': 20,

    # ==================== 信号窗口与STFT输入 ====================
    # 每张图来自4096点振动（51200Hz下约80ms）；不同样本窗口不重叠。
    # 记录长度不一，按实际长度均匀取窗，容量不足直接报错，不补零凑数量。
    'window_size': 4096,
    'img_size': 64,        # STFT缩放后的高和宽，输入张量为[N,1,64,64]。
    'in_channels': 1,      # 单振动通道生成单通道灰度图。
    # demean：每个窗口减去自身均值；zscore：再除以自身标准差。
    # 不使用整个目标数据池的统计量归一化。
    'normalize': 'demean',
    'stft_fs': 51200,      # 振动采样率Hz；原MAT的fs字段表示轴频，不能填到这里。
    'stft_nperseg': 256,   # STFT每个短时分析窗的点数，区别于4096点样本窗口。
    'stft_noverlap': 192,  # 同一张图内部相邻STFT短窗重叠192点，步长64点。
                          # 这不表示训练/验证/测试之间的样本窗口重叠。
    'stft_window': 'hann',
    'stft_bandpass': None, # None不做带通；设为[低频Hz,高频Hz]时逐样本滤波。
    'stft_log_eps': 1e-8,  # 计算log10幅值谱前加的小量，避免log(0)。
    'stft_p_low': 5,       # 幅值谱按每张图的第5/95百分位截断并拉伸。
    'stft_p_high': 95,
    'stft_gamma': 0.7,     # 拉伸后的gamma变换；小于1可增强较弱的谱结构。

    # ==================== 模型骨干（configured_maml生效） ====================
    'backbone': 'lsk_lite',             # 当前最终骨干；标准maml自动改为cnn4。
    'hidden_size': 64,                 # CNN4的通道宽度，LSK-lite使用下方stage_channels。
    'lsk_stage_channels': (32, 64, 96), # 三个阶段的输出通道数。
    'lsk_stage_depths': (1, 1, 1),      # 三个阶段各自堆叠的LSK block数量。
    'lsk_mlp_ratio': 2,                # block内MLP隐藏通道相对输入通道的扩展倍数。
    'frequency_module': 'none',        # 当前不启用额外频域模块。
    'denoise_module': 'none',          # 当前不启用额外去噪模块。
    'attention_module': 'gcnet',       # 全局上下文模块；none用于关闭该模块的消融。
    'gcnet_ratio': 0.125,              # 上下文变换瓶颈通道比例。
    'gcnet_pooling_type': 'att',       # att：学习注意力汇聚；avg：全局平均汇聚。
    'gcnet_fusion_types': ('channel_mul',), # 采用通道乘性融合；单元素元组需保留逗号。
    'gcnet_layer_scale_init': 1e-4,    # 上下文分支的可学习缩放系数初始值。

    # ==================== GGM-RW元任务加权 ====================
    # ggm_rw：根据pilot得到的任务差距加权；none：等权MAML。
    # 若仅比较GGM增益，可保留configured_maml和骨干，将此项改为none。
    'task_weighting_mode': 'ggm_rw',
    'task_weight_temperature': 1.0,    # softmax温度，越小越集中于较大的标准化差距。
    'task_weight_alpha_max': 0.20,     # 难度权重与均匀权重混合时，难度项的最终占比。
    'task_weight_warmup_epochs': 50,   # 前50次外循环更新逐步增大难度项占比。
    'task_weight_z_clip': 2.0,         # 标准化差距截断在[-2,2]内，抑制极端任务。
    'task_weight_max_ratio': 3.0,      # 单任务权重上限为均匀权重的3倍，权重总和仍为1。
    'task_weight_eps': 1e-6,           # 标准化和权重归一化的数值稳定小量。

    # ==================== 元训练与少样本适应 ====================
    'inner_lr': 0.05,                 # 在每个episode的support上适应的学习率。
    'outer_lr': 0.005,                # 源任务query损失更新基础模型时的Adam初始学习率。
    'epochs': 500,                    # 最多500次outer update，不是完整遍历数据池500次。
    'meta_batch_size': 16,            # 每次outer update汇总16个源训练episode。
    'outer_lr_min': 5e-5,             # 当前入口固定使用cosine调度，此为最低学习率。
    'outer_grad_clip_norm': 1.0,      # 对汇总后的外循环梯度做范数裁剪。
    'adaptation_steps': {1: 3, 5: 3},  # 键是shot；值是训练及validation时的support适应步数。
    'test_inner_steps': 10,           # 最终test每episode的support适应步数。

    # ==================== 目标域验证与提前停止 ====================
    # 每次outer update后，使用同一组固定目标验证episode评分。
    # 按验证准确率保存best；验证损失不回传到基础模型的元参数。
    'validation_episodes': 16,
    'validation_episode_seed': 20260913, # 专门用于固定验证清单，不受训练seed影响。
    'early_stop_on_perfect_validation': True, # False则训练满epochs；--quick时不启用早停。
    'early_stop_trigger_accuracy': 1.0,      # 常规验证达到100%后，才启动额外确认。
    'early_stop_confirmation_episodes': 100,# 额外固定验证任务数，与常规验证任务分开取。
    'early_stop_confirmation_accuracy': 0.99,# 确认准确率达到99%才提前结束训练。

    # ==================== 最终测试与兼容字段 ====================
    # 这里是测试episode总数，不是一次送入网络的图片数；每episode仍为75个query。
    # 同一池的图片可以在不同episode复用，不能解释成1000个独立物理轴承。
    'test_meta_batch_size': 1000,
    'test_episode_seed': 20260914, # 固定测试清单，跨方法/训练seed共享，1/5-shot配对query。
    # 以下两个字段保留与公共配置的兼容性，当前HUST入口不使用它们控制动作/画图：
    # 无动作参数默认训练+测试；仅测试必须传--test；仅预处理传--preprocess。
    'run_mode': 'train_test',
    # 当前HUST入口直接保存JSON历史/指标和CSV预测，不调用公共可视化流程。
    'save_visualizations': False,
}


def resolve_hust_config(overrides=None):
    """合并覆盖项、检查当前协议，并生成域划分、图片路径及权重名称。"""
    from classic_config import resolve_method_config
    # 深拷贝避免一次运行修改列表/字典后影响下一个任务。
    cfg = deepcopy(HUST_CONFIG)
    cfg.update(overrides or {})
    if cfg['task'] not in TASKS:
        raise ValueError(f'Unknown HUST task: {cfg["task"]}')
    if cfg['method'] not in ('configured_maml', 'maml'):
        raise ValueError('HUST entry currently supports configured_maml and maml')
    if cfg['n_way'] != 5 or cfg['k_shot'] not in (1, 5) or cfg['q_query'] != 15:
        raise ValueError('HUST protocol requires 5-way, 1/5-shot and 15 queries/class')
    for key in ('epochs', 'meta_batch_size', 'validation_episodes', 'test_meta_batch_size'):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    # 以下是派生字段：由task自动生成，无须再手动填source_condition/target_condition。
    source, target = TASKS[cfg['task']]
    cfg['source_condition'], cfg['target_condition'] = list(source), target
    cfg['root_path'] = str(Path(cfg['processed_root']).resolve() / cfg['task'])
    # 标准maml在这里强制使用CNN4并关闭增强模块；configured_maml保留下方自定义设置。
    cfg = resolve_method_config(cfg)
    # 名称区分任务/方法/shot/seed；--quick另外追加_quick，best文件再追加_best。
    # 骨干细节、学习率和GGM消融不会自动进入名称，另跑这些设置请指定新的--model-path。
    cfg['model_name'] = (f"HUST_{cfg['task']}_{cfg['method']}_"
                         f"5w{cfg['k_shot']}s_seed{cfg['seed']}")
    return cfg
