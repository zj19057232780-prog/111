"""跨数据集专用配置；不修改或读取原PU/CWRU/HUST的可变配置。"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
from classic_config import resolve_method_config

BASE = Path(__file__).resolve().parent
TASKS = {'T1': ('A', 'B'), 'T2': ('B', 'A'), 'T3': ('A', 'C'),
         'T4': ('C', 'A'), 'T5': ('B', 'C'), 'T6': ('C', 'B')}
METHODS = ('configured_maml', 'maml', 'protonet', 'tl_wdcnn')
CROSS_CONFIG = {
    # 修改task/method/k_shot/seed/run_mode即可切换；CLI仅覆盖显式传入值。
    'task': 'T2', 'method': 'configured_maml', 'k_shot': 1, 'seed': 24,
    'run_mode': 'train_test',
    'cwru_raw_root': str(BASE / 'CWRU'),
    'pu_raw_root': r'C:\数据集\德国帕德博恩轴承数据集',
    'hust_raw_root': str(BASE / 'HUST bearing a practical dataset for ball bearing fault diagnosis'),
    'processed_root': str(BASE / 'cross_data_processed'),
    'checkpoint_root': str(BASE / 'model_save' / 'cross_dataset'),
    'result_root': str(BASE / 'results' / 'cross_dataset'),
    'protocol': 'Cross-ABC-3way-targetval-v1',
    'class_names': ['N', 'I', 'O'], 'n_way': 3, 'q_query': 15,
    # 每域每类固定40窗口。作为目标时前20验证、后20测试；作为源时合并。
    'samples_per_block': 20, 'pu_record': 1,
    'stft_fs': 48000, 'window_size': 3072, 'img_size': 64, 'in_channels': 1,
    'normalize': 'demean', 'stft_nperseg': 192, 'stft_noverlap': 144,
    'stft_window': 'hann', 'stft_bandpass': None, 'stft_log_eps': 1e-8,
    'stft_p_low': 5, 'stft_p_high': 95, 'stft_gamma': 0.7,
    # configured_maml使用以下SCG-MAML设计；经典模型自动关闭增强和加权。
    'backbone': 'lsk_lite', 'hidden_size': 64,
    'lsk_stage_channels': (32, 64, 96), 'lsk_stage_depths': (1, 1, 1),
    'lsk_mlp_ratio': 2, 'frequency_module': 'none', 'denoise_module': 'none',
    'attention_module': 'gcnet', 'gcnet_ratio': .125, 'gcnet_pooling_type': 'att',
    'gcnet_fusion_types': ('channel_mul',), 'gcnet_layer_scale_init': 1e-4,
    'task_weighting_mode': 'ggm_rw', 'task_weight_temperature': 1.,
    'task_weight_alpha_max': .2, 'task_weight_warmup_epochs': 50,
    'task_weight_z_clip': 2., 'task_weight_max_ratio': 3., 'task_weight_eps': 1e-6,
    'metric_temperature': 1.,
    'inner_lr': .05, 'outer_lr': .005, 'epochs': 500, 'meta_batch_size': 16,
    'outer_lr_min': 5e-5, 'outer_grad_clip_norm': 1.,
    'adaptation_steps': {1: 3, 5: 3}, 'test_inner_steps': 10,
    # 目标验证标签参与选模；不表示完全未见目标域。
    'validation_episodes': 16, 'validation_episode_seed': 20260914,
    'early_stop_on_perfect_validation': True, 'early_stop_trigger_accuracy': 1.,
    'early_stop_confirmation_episodes': 100, 'early_stop_confirmation_accuracy': .99,
    'test_meta_batch_size': 1000, 'test_episode_seed': 20260915,
    # TL-WDCNN：源域监督预训练；每个目标episode重新初始化局部头并仅用support微调。
    'ft_pretrain_epochs': 100, 'ft_pretrain_batch_size': 32,
    'ft_pretrain_lr': .001, 'ft_weight_decay': 1e-4,
    'ft_steps': 10, 'ft_lr': .01, 'ft_scope': 'all', 'ft_bn_mode': 'batch',
    'ft_pretrain_max_batches': 0,
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def resolve_cross_config(overrides=None):
    cfg = deepcopy(CROSS_CONFIG)
    cfg.update(overrides or {})
    cfg['method'] = str(cfg['method']).strip().lower()
    if cfg['task'] not in TASKS or cfg['method'] not in METHODS:
        raise ValueError('Unknown cross-dataset task or method')
    if cfg['n_way'] != 3 or cfg['class_names'] != ['N', 'I', 'O'] or cfg['k_shot'] not in (1, 5):
        raise ValueError('Cross protocol requires N/I/O, 3-way, 1/5-shot')
    positive = ('q_query', 'samples_per_block', 'stft_fs', 'window_size', 'img_size',
                'epochs', 'meta_batch_size', 'validation_episodes', 'test_meta_batch_size',
                'test_inner_steps', 'early_stop_confirmation_episodes', 'pu_record')
    for key in positive:
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['q_query'] + 5 > cfg['samples_per_block']:
        raise ValueError('Target pool must hold 5 support candidates + query samples')
    if not 0 <= cfg['stft_noverlap'] < cfg['stft_nperseg'] <= cfg['window_size']:
        raise ValueError('Invalid STFT window/overlap')
    if cfg['normalize'] not in ('demean', 'zscore'):
        raise ValueError('normalize must be demean or zscore')
    cfg['source_domain'], cfg['target_domain'] = TASKS[cfg['task']]
    cfg['source_condition'], cfg['target_condition'] = [cfg['source_domain']], cfg['target_domain']
    cfg = resolve_method_config(cfg)
    # Evaluation counts, paths and action selection do not change checkpoint identity.
    exclude = {'run_mode', 'model_name', 'test_meta_batch_size', 'test_episode_seed',
               'processed_root', 'checkpoint_root', 'result_root',
               'cwru_raw_root', 'pu_raw_root', 'hust_raw_root'}
    cfg['experiment_signature'] = fingerprint({k: v for k, v in cfg.items() if k not in exclude})
    cfg['model_name'] = (f"Cross_{cfg['task']}_{cfg['method']}_3w{cfg['k_shot']}s_"
                         f"seed{cfg['seed']}_{cfg['experiment_signature'][:10]}")
    return cfg
