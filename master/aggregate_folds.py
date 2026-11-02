"""Aggregate fold 5-9 predictions and reproduce the paper's mean ROC report."""

import argparse
import json
import os
import tempfile

os.environ.setdefault('MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'dsss-fassm-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from test_Luna import classification_metrics


PAPER_FOLDS = tuple(range(5, 10))


def parse_prediction_spec(value):
    try:
        fold_text, path = value.split('=', 1)
        fold = int(fold_text)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('prediction must use FOLD=/path/to/predictions.npz')
    if fold not in PAPER_FOLDS:
        raise argparse.ArgumentTypeError('fold must be one of 5, 6, 7, 8, 9')
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError('prediction file does not exist: {}'.format(path))
    return fold, path


def aggregate_predictions(prediction_paths, threshold=0.5, grid_size=1001):
    fpr_grid = np.linspace(0.0, 1.0, grid_size)
    fold_metrics = {}
    fold_curves = {}
    interpolated_tprs = []

    for fold in PAPER_FOLDS:
        payload = np.load(prediction_paths[fold])
        targets = np.asarray(payload['targets'], dtype=np.int64)
        scores = np.asarray(payload['scores'], dtype=np.float64)
        if targets.shape != scores.shape or targets.ndim != 1:
            raise ValueError('fold{} targets and scores must be matching 1D arrays'.format(fold))
        metrics = classification_metrics(targets, scores, threshold)
        if metrics['auc'] is None:
            raise ValueError('fold{} must contain both classes for ROC/AUC'.format(fold))
        fpr, tpr, _ = roc_curve(targets, scores)
        interp_tpr = np.interp(fpr_grid, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interpolated_tprs.append(interp_tpr)
        fold_metrics[str(fold)] = metrics
        fold_curves[fold] = (fpr, tpr)

    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    auc_values = np.asarray([fold_metrics[str(fold)]['auc'] for fold in PAPER_FOLDS])
    metric_names = ('accuracy', 'sensitivity', 'specificity', 'f1')
    summary = {
        'folds': fold_metrics,
        'threshold': float(threshold),
        'mean_metrics': {
            name: float(np.mean([fold_metrics[str(fold)][name] for fold in PAPER_FOLDS]))
            for name in metric_names
        },
        'mean_fold_auc': float(np.mean(auc_values)),
        'std_fold_auc': float(np.std(auc_values)),
        'mean_roc_auc': float(auc(fpr_grid, mean_tpr)),
    }
    return summary, fold_curves, fpr_grid, mean_tpr


def save_roc_plot(path, fold_curves, fpr_grid, mean_tpr, summary):
    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    for fold in PAPER_FOLDS:
        fpr, tpr = fold_curves[fold]
        axis.plot(fpr, tpr, linewidth=1.0,
                  label='Fold {} (AUC = {:.3f})'.format(fold, summary['folds'][str(fold)]['auc']))
    axis.plot(
        fpr_grid, mean_tpr, color='blue', linewidth=2.2,
        label='Mean ROC (AUC = {:.3f} ± {:.3f})'.format(
            summary['mean_roc_auc'], summary['std_fold_auc']
        ),
    )
    axis.plot([0, 1], [0, 1], '--', color='tomato', linewidth=1.0)
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel='False Positive Rate',
             ylabel='True Positive Rate', title='ROC Curves Across Folds 5-9')
    axis.grid(alpha=0.25)
    axis.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Aggregate FASSM fold 5-9 predictions')
    parser.add_argument('--prediction', action='append', required=True, type=parse_prediction_spec,
                        help='repeat exactly once per fold: FOLD=/path/to/predictions.npz')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output_dir', default='outputs/five-fold')
    args = parser.parse_args()
    prediction_paths = dict(args.prediction)
    if set(prediction_paths) != set(PAPER_FOLDS) or len(args.prediction) != len(PAPER_FOLDS):
        parser.error('--prediction must provide each fold 5-9 exactly once')

    summary, fold_curves, fpr_grid, mean_tpr = aggregate_predictions(
        prediction_paths, args.threshold
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'five_fold_metrics.json'), 'w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
    np.savetxt(
        os.path.join(args.output_dir, 'mean_roc.csv'),
        np.column_stack([fpr_grid, mean_tpr]),
        delimiter=',', header='fpr,mean_tpr', comments='',
    )
    save_roc_plot(
        os.path.join(args.output_dir, 'roc_folds_5_9.png'),
        fold_curves, fpr_grid, mean_tpr, summary,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
