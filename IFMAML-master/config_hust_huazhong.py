"""Huazhong HUSTbearing only; never the retired Hanoi dataset."""
from copy import deepcopy
from pathlib import Path
from config_pu import PU_CONFIG, resolve_pu_config
from run_table_row import configure

ROOT = Path(__file__).resolve().parent
CONDITIONS = (20, 40, 60, 80)
TASK_PAIRS = {
    'H1': (20, 40), 'H2': (40, 20),
    'H3': (40, 60), 'H4': (60, 40),
    'H5': (60, 80), 'H6': (80, 60),
}
TASKS = {task: target for task, (source, target) in TASK_PAIRS.items()}
CLASSES = ('healthy', 'inner_medium', 'inner_severe', 'outer_medium', 'outer_severe',
           'ball_medium', 'ball_severe', 'combo_medium', 'combo_severe')
PREFIXES = dict(zip(CLASSES, ('H', '0.5X_I', 'I', '0.5X_O', 'O', '0.5X_B', 'B', '0.5X_C', 'C')))
MODELS = {'cnn4': 'cnn4maml', 'lsk': 'lskmaml', 'lsk_gc': 'lskgcmaml', 'full': 'final'}
BASELINES = ('protonet', 'matchingnet', 'relationnet', 'maml', 'tl_wdcnn', 'amaml', 'tubr_ml', 'mfeml')
PAPERS = {'amaml': 'mu2025_amaml', 'tubr_ml': 'wang2025_tubr_ml', 'mfeml': 'li2025_mfeml'}
MODELS.update({name:name for name in BASELINES})
SEEDS = (3, 24, 38)
# Cache identity is unchanged: it stores all conditions, independent of task routing.
CACHE_PROTOCOL = 'HUST-Huazhong-9pool-5way-targetval-v1'
PROTOCOL = 'HUST-Huazhong-9pool-5way-single-source-targetval-v2'


def configuration(task, model, seed, shots=1, channel='X', input_mode='stft', patience=100):
    if task not in TASKS or model not in MODELS or shots not in (1, 3, 5) or channel not in ('X', 'Y', 'Z'):
        raise ValueError('Invalid HUST task/model/shot/channel')
    if model == 'tubr_ml' and shots == 3:
        raise ValueError('TUBR-ML is excluded from HUST 3-shot experiments')
    if input_mode not in ('stft','nostft') or patience < 0:
        raise ValueError('Invalid input/patience')
    if input_mode=='nostft' and model in ('full','lsk','lsk_gc','cnn4'):
        raise ValueError('SCG-MAML/ablation presets support STFT only; use maml for native CNN4')
    if model=='tubr_ml' and input_mode=='nostft' and shots==5:
        raise ValueError('TUBR-ML nostft 5-shot + 15-query exceeds the disjoint target pool; use STFT or 1-shot')
    # Reuse exactly the existing four model presets and PU learning parameters.
    base = resolve_pu_config(deepcopy(PU_CONFIG))
    cfg = configure(base, 'cnn4maml' if model in PAPERS else MODELS[model], 'N09_M07_F10', 'pu', shots, seed, 500)
    cfg.update(dataset='hust_huazhong', protocol=PROTOCOL, task=task, model_variant=model,
        root_path=str(ROOT/'hust_huazhong_processed'/channel),
        raw_root=str(ROOT/'HUST-bearing-dataset'), channel=channel,
        source_condition=[str(TASK_PAIRS[task][0])],
        target_condition=str(TASKS[task]), classification_preset='huazhong_9state',
        class_groups={c:[c] for c in CLASSES}, class_names=list(CLASSES),
        window_size=4096, img_size=64, stft_fs=25600, stft_nperseg=256, stft_noverlap=192,
        target_validation_fraction=.5, split_gap_windows=1, validation_domain='target',
        epochs=500, meta_batch_size=16, test_meta_batch_size=1000,
        # 所有HUST模型（含full）共用验证准确率patience；0关闭，默认100轮。
        early_stop_on_perfect_validation=False, early_stop_patience=patience,
        ft_patience=patience, ft_pretrain_epochs=100, validate_after_update=True,
        tsne_selected_classes=[], input_mode=input_mode,
        input_representation='stft' if input_mode=='stft' else 'raw_1d', smoke=False)
    return cfg
