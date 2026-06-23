"""Independent CWRU configuration for cross-condition few-shot diagnosis."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


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


CWRU_CONDITIONS = {
    'C0_1797rpm': '1797',
    'C1_1772rpm': '1772',
    'C2_1750rpm': '1750',
    'C3_1730rpm': '1730',
}


CWRU_CONFIG = {
    # Data paths are independent from the existing PU data directories.
    'raw_data_root': str(PROJECT_ROOT / 'CWRU'),
    'root_path': str(PROJECT_ROOT / 'cwru_data_processed'),

    # Three seen load conditions are merged for meta-training. The 3 HP
    # condition is held out for validation and testing.
    'condition_speeds': dict(CWRU_CONDITIONS),
    'source_condition': ['C0_1797rpm', 'C1_1772rpm', 'C2_1750rpm'],
    'target_condition': 'C3_1730rpm',
    'classification_preset': 'cwru_10state_3source_1target',
    'class_specs': dict(CWRU_CLASS_SPECS),
    'class_names': list(CWRU_CLASS_SPECS),

    # CWRU 48 kHz drive-end signals. These values preserve the same physical
    # time scales as the PU setup: 64 ms samples, 4 ms STFT windows, 1 ms hops.
    'preprocess_mode': 'stft_log',
    'signal_channel': 'DE_time',
    'stft_fs': 48000,
    'window_size': 3072,
    'stft_nperseg': 192,
    'stft_noverlap': 144,
    'stft_window': 'hann',
    'stft_bandpass': None,
    'stft_log_eps': 1e-8,
    'stft_p_low': 5,
    'stft_p_high': 95,
    'stft_gamma': 0.7,
    'normalize': 'demean',
    'img_size': 64,
    'in_channels': 1,
    'source_samples_per_class': 40,
    'target_val_samples_per_class': 20,
    'target_test_samples_per_class': 20,
    'preprocess_seed': 24,

    # Keep a documented exception for the malformed downloaded archive.
    # metadata.txt points to X174, but this file contains X173 at 1796 rpm.
    'variable_overrides': {
        '48DriveEndFault/1797/0.014-InnerRace.mat': 'X173_DE_time',
    },

    # Model settings are separate from config_pu.py, so CWRU ablations cannot
    # silently change the existing PU experiment.
    'backbone': 'lsk_lite',
    'model_name': 'CWRU_STFT_LSKLite_ARSM_MAML_10state_C012_to_C3',
    'hidden_size': 64,
    'lsk_stage_channels': (32, 64, 96),
    'lsk_stage_depths': (1, 1, 1),
    'lsk_mlp_ratio': 2,
    'frequency_module': 'none',
    'gfnet_depth': 1,
    'gfnet_mlp_ratio': 2,
    'gfnet_weight_scale': 0.02,
    'gfnet_layer_scale_init': 1e-2,
    'denoise_module': 'arsm',
    'arsm_reduction': 4,
    'arsm_blend_init': 1e-3,
    'attention_module': 'none',
    'ema_factor': 8,
    'ema_layer_scale_init': 1e-3,
    'gcnet_ratio': 0.125,
    'gcnet_pooling_type': 'att',
    'gcnet_fusion_types': ('channel_mul',),
    'gcnet_layer_scale_init': 1e-4,

    # Match the current PU episodic protocol.
    'n_way': 5,
    'k_shot': 1,
    'q_query': 15,
    'inner_lr': 0.05,
    'outer_lr': 0.005,
    'outer_lr_scheduler': 'cosine',
    'outer_lr_min': 5e-5,
    'outer_grad_clip_norm': 1.0,
    'epochs': 500,
    'tasks_per_epoch': 1000,
    'meta_batch_size': 16,
    'test_meta_batch_size': 1000,
    'test_inner_steps': 10,
    'adaptation_steps': {
        1: 3,
        5: 1,
    },
    'seed': 24,

    'visualization_dir': str(PROJECT_ROOT / 'results' / 'cwru'),
    'save_visualizations': True,
    'tsne_max_points': 1000,
    'tsne_perplexity': 30,
    'tsne_selected_classes': [
        'Healthy',
        'IR_007',
        'IR_021',
        'Ball_014',
        'OR_021',
    ],
}


QUICK_TEST = False
