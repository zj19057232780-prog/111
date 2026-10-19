"""跨数据集专用配置；不修改或读取原PU/CWRU/HUST的可变配置。"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import hashlib
import json


BASE = Path(__file__).resolve().parent
CWRU_CONDITIONS = {'C0': '1797', 'C1': '1772', 'C2': '1750', 'C3': '1730'}
PU_CONDITIONS = {'P1': 'N09_M07_F10', 'P2': 'N15_M01_F10',
                 'P3': 'N15_M07_F04', 'P4': 'N15_M07_F10'}
TASKS = {**{f'T{i+1}': ('A', 'B', c) for i, c in enumerate(PU_CONDITIONS)},
         **{f'T{i+5}': ('B', 'A', c) for i, c in enumerate(CWRU_CONDITIONS)}}
TASK_GROUPS = {'CWRU_to_PU': tuple(f'T{i}' for i in range(1, 5)),
               'PU_to_CWRU': tuple(f'T{i}' for i in range(5, 9))}
METHODS = ('protonet', 'matchingnet', 'relationnet', 'maml', 'tl_wdcnn',
           'amaml', 'tubr_ml', 'mfeml', 'configured_maml')
CLASS_NAMES = ['N', 'I-L1', 'I-L2', 'O-L1', 'O-L2']
METHOD_ALIASES = {'scg_maml': 'configured_maml', 'scgmaml': 'configured_maml',
                  'tubrml': 'tubr_ml', 'tlwdcnn': 'tl_wdcnn'}
CROSS_CONFIG = {
    # 修改task/method/k_shot/seed/run_mode即可切换；CLI仅覆盖显式传入值。
    'task': 'T2', 'method': 'configured_maml', 'k_shot': 1, 'seed': 24,
    'run_mode': 'train_test',
    'cwru_raw_root': str(BASE / 'CWRU'),
    'pu_raw_root': r'C:\数据集\德国帕德博恩轴承数据集',
    'processed_root': str(BASE / 'cross_data_processed_ab5_ms_v2'),
    'checkpoint_root': str(BASE / 'model_save' / 'cross_dataset_ab5_ms_v2'),
    'result_root': str(BASE / 'results' / 'cross_dataset_ab5_ms_v2'),
    'protocol': 'Cross-AB-MS-5way-targetval-v2',
    'class_names': CLASS_NAMES, 'n_way': 5, 'q_query': 10,
    # C0短记录每段仅16个7056点不重叠窗；用户确认全任务统一10 query。
    'samples_per_block': 16, 'pu_records': [1, 2], 'anchor_size': 7056,
    'input_mode': 'stft', 'cwru_conditions': CWRU_CONDITIONS, 'pu_conditions': PU_CONDITIONS,
    'source_episode_sampling': 'uniform_condition',
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
    'adaptation_steps': {1: 3, 3: 3, 5: 3}, 'test_inner_steps': 10,
    'validation_inner_steps': 10, 'validation_interval': 10, 'ft_validation_interval': 2,
    # 目标验证标签参与选模；不表示完全未见目标域。
    'validation_episodes': 100, 'validation_episode_seed': 20261016,
    'early_stop_on_perfect_validation': False,
    # Comparison models: epochs since strict validation-accuracy improvement.
    # SCG-MAML is always resolved to 0 (fixed training budget).
    'early_stop_patience': 100,
    'test_meta_batch_size': 1000, 'test_episode_seed': 20261017,
    # TL-WDCNN：源域监督预训练；每个目标episode重新初始化局部头并仅用support微调。
    'ft_pretrain_epochs': 100, 'ft_pretrain_batch_size': 32,
    'ft_pretrain_lr': .001, 'ft_weight_decay': 1e-4,
    'ft_steps': 10, 'ft_lr': .01, 'ft_scope': 'all', 'ft_bn_mode': 'batch',
    'ft_pretrain_max_batches': 0,
    'matching_fce': True, 'matching_fce_steps': 5, 'relation_hidden_size': 8,
    'raw_feature_bins': 4, 'method_options': {},
    'bn_protocol': 'episode_local_batch_query',
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def canonical_method(value: str) -> str:
    name = str(value).strip().lower().replace('-', '_').replace(' ', '')
    return METHOD_ALIASES.get(name, name)


def resolve_cross_config(overrides: dict | None = None) -> dict:
    import math
    cfg = deepcopy(CROSS_CONFIG)
    cfg.update(overrides or {})
    cfg['method'] = canonical_method(cfg['method'])
    cfg['input_mode'] = str(cfg['input_mode']).lower().replace('no_stft', 'nostft')
    if cfg['task'] not in TASKS or cfg['method'] not in METHODS:
        raise ValueError('Unknown cross-dataset task or method')
    if type(cfg['early_stop_patience']) is not int or cfg['early_stop_patience'] < 0:
        raise ValueError('early_stop_patience must be a non-negative integer')
    if cfg['method'] == 'configured_maml':
        cfg['early_stop_patience'] = 0
    if cfg['protocol'] != 'Cross-AB-MS-5way-targetval-v2':
        raise ValueError('Legacy cross-dataset protocols require their original entry point')
    if cfg['n_way'] != 5 or cfg['class_names'] != CLASS_NAMES or cfg['k_shot'] not in (1, 3, 5):
        raise ValueError('Cross AB requires five canonical classes and 1/3/5-shot')
    if cfg['input_mode'] not in ('stft', 'nostft'):
        raise ValueError('input_mode must be stft or nostft')
    if cfg['input_mode'] == 'nostft' and cfg['method'] == 'configured_maml':
        raise ValueError('SCG-MAML has no existing nostft implementation; use --input stft')
    if cfg['pu_records'] != [1, 2] or cfg['anchor_size'] != 7056:
        raise ValueError('Protocol requires PU records 1/2 and 7056-point native anchors')
    if cfg['cwru_conditions'] != CWRU_CONDITIONS or cfg['pu_conditions'] != PU_CONDITIONS:
        raise ValueError('Cross AB MS v2 requires all four conditions of each dataset')
    if cfg['source_episode_sampling'] != 'uniform_condition':
        raise ValueError('Source episodes must sample the four source conditions uniformly')
    for key in ('q_query', 'samples_per_block', 'stft_fs', 'window_size', 'img_size',
                'epochs', 'meta_batch_size', 'validation_episodes', 'test_meta_batch_size',
                'test_inner_steps', 'validation_inner_steps', 'validation_interval',
                'ft_validation_interval', 'ft_pretrain_epochs', 'ft_pretrain_batch_size', 'ft_steps'):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['q_query'] + 5 > cfg['samples_per_block']:
        raise ValueError('Target pool needs five support candidates plus queries')
    if cfg['window_size'] != 3072 or cfg['stft_fs'] != 48000 or cfg['img_size'] != 64:
        raise ValueError('STFT interface is fixed at 3072 points / 48 kHz / 64x64')
    if not 0 <= cfg['stft_noverlap'] < cfg['stft_nperseg'] <= cfg['window_size']:
        raise ValueError('Invalid STFT overlap')
    for key in ('inner_lr', 'outer_lr', 'outer_lr_min', 'outer_grad_clip_norm',
                'ft_pretrain_lr', 'ft_lr', 'metric_temperature'):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if cfg['run_mode'] not in ('train', 'test', 'train_test'):
        raise ValueError('Invalid run_mode')
    if cfg['early_stop_on_perfect_validation']:
        raise ValueError('Cross AB MS v2 does not use perfect-validation early stop')
    if cfg['ft_bn_mode'] != 'batch' or cfg['ft_scope'] != 'all':
        raise ValueError('Cross AB uses full-model support adaptation and episode batch BN')
    if cfg['ft_steps'] != cfg['test_inner_steps'] or cfg['validation_inner_steps'] != cfg['test_inner_steps']:
        raise ValueError('Validation/test adaptation budgets must match (including TL-WDCNN)')
    cfg['source_domain'], cfg['target_domain'], cfg['target_condition'] = TASKS[cfg['task']]
    cfg['input_representation'] = cfg['input_mode']
    cfg['source_conditions'] = list(CWRU_CONDITIONS if cfg['source_domain']=='A' else PU_CONDITIONS)
    cfg['source_condition'] = list(cfg['source_conditions'])
    target_map = PU_CONDITIONS if cfg['target_domain']=='B' else CWRU_CONDITIONS
    cfg['target_condition_name'] = target_map[cfg['target_condition']]
    if cfg['method'] != 'configured_maml':
        cfg.update(backbone='cnn4', attention_module='none', task_weighting_mode='none')
    # Reuse each reconstruction's actual method defaults; include them in identity.
    if cfg['method'] in ('amaml', 'tubr_ml', 'mfeml'):
        from dataclasses import asdict
        from cross_dataset_models import paper_module
        cfg['method_options'] = asdict(paper_module(cfg['method']).MethodConfig(**cfg['method_options']))
    cfg['native_input'] = {
        'protonet': '1d zscore: CWRU3072/PU4096',
        'matchingnet': '1d zscore: CWRU3072/PU4096',
        'relationnet': '1d zscore: CWRU3072/PU4096',
        'maml': '1d zscore: CWRU3072/PU4096',
        'tl_wdcnn': '1d zscore: CWRU3072/PU4096',
        'amaml': 'raw1444 -> 1x38x38; no normalization',
        'tubr_ml': 'raw7056 -> minmax -> 3x84x84',
        'mfeml': 'raw2000 -> native AdaptiveSquare45; no normalization',
        'configured_maml': 'unsupported',
    }[cfg['method']] if cfg['input_mode'] == 'nostft' else (
        'STFT64 flattened to 4096 for existing 1D WDCNN'
        if cfg['method'] == 'tl_wdcnn' else 'STFT 1x64x64')
    excluded = {'run_mode', 'model_name', 'test_meta_batch_size', 'test_episode_seed',
                'processed_root', 'checkpoint_root', 'result_root', 'cwru_raw_root', 'pu_raw_root'}
    cfg['experiment_signature'] = fingerprint({k: v for k, v in cfg.items()
                                              if k not in excluded | {'experiment_signature'}})
    cfg['model_name'] = (f"CrossABMS_{cfg['task']}_{cfg['method']}_{cfg['input_mode']}_"
                         f"5w{cfg['k_shot']}s_seed{cfg['seed']}_{cfg['experiment_signature'][:10]}")
    return cfg


