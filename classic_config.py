"""Pure configuration resolver; importing configs never imports torch."""
import hashlib
import json
import math
METHODS = ('configured_maml', 'maml', 'protonet', 'matchingnet', 'relationnet', 'resnet18_ft', 'tl_wdcnn', 'resnet18')


def resolve_method_config(config):
    cfg = dict(config)
    method = str(cfg.get('method', 'configured_maml')).strip().lower()
    if method not in METHODS:
        raise ValueError(f'Unknown method {method!r}; choose from {METHODS}')
    cfg['method'] = method
    cfg['run_mode'] = str(cfg.get('run_mode', 'train_test')).strip().lower()
    if cfg['run_mode'] not in ('train', 'test', 'train_test'):
        raise ValueError('run_mode must be train, test or train_test')
    if method != 'configured_maml':
        cfg.update(backbone='cnn4', frequency_module='none', denoise_module='none',
                   attention_module='none', task_weighting_mode='none')
        if method in ('resnet18_ft', 'tl_wdcnn'):
            cfg['backbone'] = 'resnet18' if method == 'resnet18_ft' else 'wdcnn'
            cfg['input_representation'] = 'stft' if method == 'resnet18_ft' else 'raw_1d'
            if cfg.get('ft_scope', 'all') not in ('all', 'head'):
                raise ValueError('ft_scope must be all or head')
            if cfg.get('ft_bn_mode', 'batch') not in ('batch', 'frozen'):
                raise ValueError('ft_bn_mode must be batch or frozen')
            for key, default in [('ft_steps', 10), ('ft_pretrain_epochs', 100),
                                 ('ft_pretrain_batch_size', 64), ('ft_validation_episodes', 16),
                                 ('ft_lr', .01), ('ft_pretrain_lr', .001)]:
                value = float(cfg.get(key, default))
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f'{key} must be positive')
                if key not in ('ft_lr', 'ft_pretrain_lr') and not value.is_integer():
                    raise ValueError(f'{key} must be an integer')
            for key in ('ft_pretrain_max_batches', 'ft_patience'):
                value = float(cfg.get(key, 0))
                if not math.isfinite(value) or value < 0 or not value.is_integer():
                    raise ValueError(f'{key} must be a non-negative integer')
            if int(cfg.get('in_channels', 1)) != 1:
                raise ValueError('Transfer baselines currently require single-channel inputs')
            if method == 'tl_wdcnn' and int(cfg.get('window_size', 0)) < 512:
                raise ValueError('TL-WDCNN requires window_size >= 512')
        if method == 'resnet18':
            cfg.update(backbone='resnet18', input_representation='stft')
            if int(cfg.get('in_channels', 1)) != 1:
                raise ValueError('ResNet18 currently requires single-channel STFT')
            for key, default in [('supervised_epochs', 100), ('supervised_batch_size', 64),
                                 ('supervised_lr', .001), ('supervised_weight_decay', .0001),
                                 ('supervised_max_batches', 0)]:
                value = float(cfg.get(key, default))
                minimum_ok = value >= 0 if key in ('supervised_max_batches', 'supervised_weight_decay') else value > 0
                if not math.isfinite(value) or not minimum_ok:
                    raise ValueError(f'Invalid {key}')
                if key in ('supervised_epochs', 'supervised_batch_size', 'supervised_max_batches') and not value.is_integer():
                    raise ValueError(f'{key} must be an integer')
        # Isolate algorithms, domains, shots and seeds from existing checkpoints.
        dataset = 'PU' if 'class_groups' in cfg else 'CWRU'
        source = cfg.get('source_condition', [])
        source = [source] if isinstance(source, str) else source
        signature = {k: cfg.get(k) for k in (
            'source_condition', 'target_condition', 'classification_preset',
            'class_groups', 'class_names', 'hidden_size', 'img_size', 'in_channels',
            'matching_fce', 'matching_fce_steps', 'relation_hidden_size',
            'metric_temperature', 'q_query')}
        if method in ('resnet18_ft', 'tl_wdcnn'):
            signature.update({k: v for k, v in cfg.items() if k.startswith('ft_')})
            signature.update({k: cfg.get(k) for k in ('window_size', 'preprocess_seed', 'target_val_ratio')})
        if method == 'resnet18':
            signature.update({k: v for k, v in cfg.items() if k.startswith('supervised_')})
        tag = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:8]
        backbone_tag = 'CNN4' if cfg['backbone'] == 'cnn4' else cfg['backbone']
        cfg['model_name'] = (f"{dataset}_{method}_{backbone_tag}_{cfg.get('target_condition', 'target')}_"
                             f"{cfg['n_way']}w{cfg['k_shot']}s_seed{cfg.get('seed', 24)}_{tag}")
        if method == 'resnet18':
            pool = cfg.get('class_groups', cfg.get('class_names', []))
            cfg['model_name'] = (f"{dataset}_resnet18_{cfg.get('target_condition', 'target')}_"
                                 f"{len(pool)}classes_seed{cfg.get('seed', 24)}_{tag}")
    return cfg
