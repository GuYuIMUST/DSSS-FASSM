"""Evaluate one fold-specific FASSM checkpoint on its held-out LUNA16 subset."""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve

import transforms
from checkpointing import load_fassm_checkpoint
from dataloader import lunanod
from repro_utils import seed_everything


CROP_SIZE = 32
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a fold-specific FASSM checkpoint')
    parser.add_argument('--crop_dir', required=True,
                        help='directory containing preprocessed 32^3 nodule .npy crops')
    parser.add_argument('--luna_root', required=True,
                        help='directory containing LUNA16 subset0 through subset9')
    parser.add_argument('--annotations_path',
                        default=os.path.join(REPO_ROOT, 'data', 'annotationdetclsconvfnl_v3.csv'))
    parser.add_argument('--pretrain_model_path', required=True, help='checkpoint to evaluate')
    parser.add_argument('--fold_test', type=int, required=True, choices=range(5, 10),
                        help='held-out paper fold; must match checkpoint fold_validate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', default='outputs/evaluation')
    parser.add_argument('--init_channels', type=int, default=20)
    parser.add_argument('--layers', type=int, default=20)
    parser.add_argument('--arch', default='DARTS_MAMBA')
    parser.add_argument('--auxiliary', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--allow_legacy_pickle', action='store_true',
                        help='allow trusted legacy checkpoints that pickle the full model')
    args = parser.parse_args()

    for option, path in (
        ('--crop_dir', args.crop_dir), ('--luna_root', args.luna_root),
        ('--annotations_path', args.annotations_path),
        ('--pretrain_model_path', args.pretrain_model_path),
    ):
        if not os.path.exists(path):
            parser.error('{} must be an existing path: {}'.format(option, path))
    if not 0.0 < args.threshold < 1.0:
        parser.error('--threshold must be between 0 and 1')
    return args


def crop_statistics(crop_dir):
    paths = [
        os.path.join(crop_dir, name)
        for name in sorted(os.listdir(crop_dir)) if name.endswith('.npy')
    ]
    if not paths:
        raise ValueError('no .npy crops found in {}'.format(crop_dir))
    pixel_sum = 0.0
    pixel_count = 0
    for path in paths:
        data = np.load(path)
        pixel_sum += np.sum(data, dtype=np.float64)
        pixel_count += data.size
    mean = pixel_sum / pixel_count
    squared_sum = 0.0
    for path in paths:
        data = np.load(path)
        squared_sum += np.sum((data - mean) ** 2, dtype=np.float64)
    std = np.sqrt(squared_sum / pixel_count)
    if not np.isfinite(std) or std == 0:
        raise ValueError('invalid crop standard deviation: {}'.format(std))
    return float(mean), float(std)


def _center_crop(data):
    starts = [int(size / 2 - CROP_SIZE / 2) for size in data.shape]
    cropped = data[
        starts[0]:starts[0] + CROP_SIZE,
        starts[1]:starts[1] + CROP_SIZE,
        starts[2]:starts[2] + CROP_SIZE,
    ]
    if cropped.shape != (CROP_SIZE, CROP_SIZE, CROP_SIZE):
        raise ValueError('crop is smaller than 32^3: {}'.format(data.shape))
    return np.asarray(cropped)


def build_test_loader(args, mean, std):
    subset_dir = os.path.join(args.luna_root, 'subset{}'.format(args.fold_test))
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError('test subset directory does not exist: {}'.format(subset_dir))
    test_ids = {name[:-4] for name in os.listdir(subset_dir) if name.endswith('.mhd')}
    annotations = pd.read_csv(args.annotations_path)
    max_diameter = float(annotations['diameter_mm'].abs().max())

    file_names, labels, features = [], [], []
    grid_y, grid_x, grid_z = np.ogrid[
        -CROP_SIZE / 2:CROP_SIZE / 2,
        -CROP_SIZE / 2:CROP_SIZE / 2,
        -CROP_SIZE / 2:CROP_SIZE / 2,
    ]
    for row in annotations.itertuples(index=False):
        series_id = str(row.seriesuid)
        if series_id.split('-')[0] not in test_ids:
            continue
        crop_name = series_id + '.npy'
        crop_path = os.path.join(args.crop_dir, crop_name)
        if not os.path.isfile(crop_path):
            raise FileNotFoundError('missing nodule crop: {}'.format(crop_path))
        crop = _center_crop(np.load(crop_path))
        diameter = abs(float(row.diameter_mm))
        mask = abs(grid_y ** 3 + grid_x ** 3 + grid_z ** 3) <= diameter ** 3
        feature = np.zeros((CROP_SIZE, CROP_SIZE, CROP_SIZE), dtype=float)
        feature[mask] = 1
        if max_diameter:
            feature[-1] /= max_diameter
        file_names.append(crop)
        labels.append(int(row.malignant))
        features.append(feature)

    if not file_names:
        raise ValueError('no annotated nodules found for subset{}'.format(args.fold_test))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = lunanod(
        args.crop_dir, file_names, labels, features,
        train=False, download=True, transform=transform,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )


def infer(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    targets, scores, losses = [], [], []
    with torch.no_grad():
        for inputs, batch_targets, _ in loader:
            inputs = inputs.to(device)
            batch_targets = batch_targets.to(device)
            output = model(inputs)
            logits = output[0] if isinstance(output, tuple) else output
            losses.append((criterion(logits, batch_targets).item(), inputs.size(0)))
            scores.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            targets.extend(batch_targets.cpu().numpy().tolist())
    total = sum(count for _, count in losses)
    mean_loss = sum(loss * count for loss, count in losses) / total
    return np.asarray(targets, dtype=np.int64), np.asarray(scores, dtype=np.float64), mean_loss


def classification_metrics(targets, scores, threshold=0.5):
    predictions = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    metrics = {
        'threshold': float(threshold),
        'n': int(targets.size),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': float(100.0 * (tp + tn) / targets.size),
        'sensitivity': float(100.0 * tp / (tp + fn)) if tp + fn else None,
        'specificity': float(100.0 * tn / (tn + fp)) if tn + fp else None,
        'f1': float(100.0 * f1_score(targets, predictions, zero_division=0)),
        'auc': None,
    }
    if np.unique(targets).size == 2:
        metrics['auc'] = float(roc_auc_score(targets, scores))
    return metrics


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-capable GPU is required for FASSM evaluation')
    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda', args.gpu)

    defaults = {
        'init_channels': args.init_channels,
        'layers': args.layers,
        'arch': args.arch,
        'auxiliary': args.auxiliary,
        'num_classes': 2,
    }
    model, checkpoint_meta = load_fassm_checkpoint(
        args.pretrain_model_path, device, defaults,
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    trained_fold = checkpoint_meta['run_config'].get('fold_validate')
    if trained_fold is not None and int(trained_fold) != args.fold_test:
        raise ValueError(
            'checkpoint was trained with fold_validate={}, but fold_test={}'.format(
                trained_fold, args.fold_test
            )
        )

    mean, std = crop_statistics(args.crop_dir)
    loader = build_test_loader(args, mean, std)
    targets, scores, loss = infer(model, loader, device)
    metrics = classification_metrics(targets, scores, args.threshold)
    metrics.update({
        'fold': args.fold_test,
        'loss': float(loss),
        'normalization_mean': mean,
        'normalization_std': std,
        'checkpoint': os.path.abspath(args.pretrain_model_path),
        'checkpoint_sha256': checkpoint_meta['sha256'],
        'checkpoint_epoch': checkpoint_meta['epoch'],
        'checkpoint_valid_acc': checkpoint_meta['valid_acc'],
    })

    fold_dir = os.path.join(args.output_dir, 'fold{}'.format(args.fold_test))
    os.makedirs(fold_dir, exist_ok=True)
    with open(os.path.join(fold_dir, 'metrics.json'), 'w', encoding='utf-8') as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
    np.savez_compressed(
        os.path.join(fold_dir, 'predictions.npz'), targets=targets, scores=scores
    )
    if metrics['auc'] is not None:
        fpr, tpr, thresholds = roc_curve(targets, scores)
        np.savetxt(
            os.path.join(fold_dir, 'roc.csv'),
            np.column_stack([fpr, tpr, thresholds]),
            delimiter=',', header='fpr,tpr,threshold', comments='',
        )

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
