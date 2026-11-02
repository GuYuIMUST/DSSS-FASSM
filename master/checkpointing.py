"""Versioned FASSM checkpoint loading with legacy-checkpoint compatibility."""

import hashlib

import torch
import torch.nn as nn

import genotypes
from model import NetworkLUNA


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _config_dict(checkpoint):
    config = checkpoint.get('run_config', {}) if isinstance(checkpoint, dict) else {}
    if hasattr(config, '__dict__'):
        config = vars(config)
    return dict(config)


def load_fassm_checkpoint(path, device, defaults=None, allow_legacy_pickle=False):
    """Load a state-dict checkpoint, falling back to trusted legacy pickles."""
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        if not allow_legacy_pickle:
            raise ValueError(
                'checkpoint requires unsafe legacy pickle loading; rerun with '
                '--allow_legacy_pickle only for a checkpoint you trust'
            )
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    defaults = dict(defaults or {})
    run_config = _config_dict(checkpoint)

    legacy_model = checkpoint.get('model') if isinstance(checkpoint, dict) else None
    state_dict = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else None

    if state_dict is not None and run_config:
        config = {**defaults, **run_config}
        genotype_name = config.get('arch', 'DARTS_MAMBA')
        if not hasattr(genotypes, genotype_name):
            raise ValueError('unknown genotype in checkpoint: {}'.format(genotype_name))
        model = NetworkLUNA(
            int(config.get('init_channels', 20)),
            int(config.get('num_classes', 2)),
            int(config.get('layers', 20)),
            bool(config.get('auxiliary', True)),
            getattr(genotypes, genotype_name),
        )
        model.load_state_dict(state_dict, strict=True)
    elif isinstance(legacy_model, nn.Module):
        # Legacy checkpoints pickle the full module. Only load files produced
        # by a trusted training run.
        model = legacy_model
        config = defaults
    elif state_dict is not None:
        config = defaults
        genotype_name = config.get('arch', 'DARTS_MAMBA')
        model = NetworkLUNA(
            int(config.get('init_channels', 20)),
            int(config.get('num_classes', 2)),
            int(config.get('layers', 20)),
            bool(config.get('auxiliary', True)),
            getattr(genotypes, genotype_name),
        )
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError('unsupported checkpoint format: model_state_dict is missing')

    model = model.to(device)
    metadata = {
        'epoch': checkpoint.get('epoch') if isinstance(checkpoint, dict) else None,
        'valid_acc': checkpoint.get('valid_acc') if isinstance(checkpoint, dict) else None,
        'run_config': run_config,
        'sha256': sha256_file(path),
        'legacy_full_model': isinstance(legacy_model, nn.Module) and not run_config,
    }
    return model, metadata
