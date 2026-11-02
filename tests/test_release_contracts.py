import os
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
import yaml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MASTER_DIR = os.path.join(REPO_ROOT, 'master')
if MASTER_DIR not in sys.path:
    sys.path.insert(0, MASTER_DIR)

import genotypes
from aggregate_folds import PAPER_FOLDS, aggregate_predictions
from checkpointing import load_fassm_checkpoint
from model import NetworkLUNA
from repro_utils import capture_rng_state, restore_rng_state, seed_everything
from test_Luna import _center_crop, classification_metrics


class ReleaseContractTests(unittest.TestCase):

    def test_published_architecture_parameter_count(self):
        self.assertIs(genotypes.DARTS_MAMBA, genotypes.DARTS_MAMBA25)
        model = NetworkLUNA(20, 2, 20, True, genotypes.DARTS_MAMBA)
        inference_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if 'auxiliary' not in name
        )
        self.assertEqual(inference_parameters, 8_586_062)

    def test_published_metadata_and_readme_contract(self):
        with open(os.path.join(REPO_ROOT, 'README.md'), encoding='utf-8') as stream:
            readme = stream.read()
        self.assertIn('## Usage', readme)
        self.assertTrue(all('Fold-{}'.format(fold) in readme for fold in PAPER_FOLDS))
        self.assertIn('--arch_subset 0', readme)
        self.assertIn('master/test_Luna.py', readme)
        self.assertNotIn('test_Luna_wms.py', readme)
        self.assertNotIn('37-hour', readme)

        with open(os.path.join(REPO_ROOT, 'CITATION.cff'), encoding='utf-8') as stream:
            citation = yaml.safe_load(stream)
        self.assertEqual(
            citation['preferred-citation']['doi'],
            '10.1016/j.compmedimag.2026.102820',
        )

    def test_fixed_threshold_metrics_and_five_fold_aggregation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            prediction_paths = {}
            targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
            scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
            for fold in PAPER_FOLDS:
                path = os.path.join(temp_dir, 'fold{}.npz'.format(fold))
                np.savez_compressed(path, targets=targets, scores=scores)
                prediction_paths[fold] = path

            metrics = classification_metrics(targets, scores, threshold=0.5)
            self.assertAlmostEqual(metrics['accuracy'], 100.0)
            self.assertAlmostEqual(metrics['sensitivity'], 100.0)
            self.assertAlmostEqual(metrics['specificity'], 100.0)
            self.assertAlmostEqual(metrics['f1'], 100.0)
            self.assertAlmostEqual(metrics['auc'], 1.0)

            summary, _, _, _ = aggregate_predictions(prediction_paths)
            self.assertAlmostEqual(summary['mean_metrics']['accuracy'], 100.0)
            self.assertAlmostEqual(summary['mean_fold_auc'], 1.0)
            self.assertAlmostEqual(summary['mean_roc_auc'], 1.0, places=3)

    def test_center_crop_is_exact_and_rejects_small_inputs(self):
        volume = np.arange(36 ** 3).reshape(36, 36, 36)
        crop = _center_crop(volume)
        self.assertEqual(crop.shape, (32, 32, 32))
        self.assertEqual(crop[0, 0, 0], volume[2, 2, 2])
        with self.assertRaisesRegex(ValueError, 'smaller than'):
            _center_crop(np.zeros((31, 32, 32)))

    def test_rng_checkpoint_round_trip(self):
        seed_everything(17)
        state = capture_rng_state()
        expected = (random.random(), np.random.random(), torch.rand(3))
        restore_rng_state(state)
        actual = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_state_dict_checkpoint_loads_without_legacy_pickle(self):
        seed_everything(5)
        model = NetworkLUNA(4, 2, 4, True, genotypes.DARTS_MAMBA)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, 'checkpoint.pth.tar')
            torch.save({
                'format_version': 1,
                'model_state_dict': model.state_dict(),
                'run_config': {
                    'init_channels': 4,
                    'num_classes': 2,
                    'layers': 4,
                    'auxiliary': True,
                    'arch': 'DARTS_MAMBA',
                    'fold_validate': 5,
                },
                'epoch': 3,
                'valid_acc': 91.0,
                'rng_state': capture_rng_state(),
            }, path)

            loaded, metadata = load_fassm_checkpoint(path, torch.device('cpu'))
            self.assertFalse(metadata['legacy_full_model'])
            self.assertEqual(metadata['epoch'], 3)
            self.assertEqual(metadata['run_config']['fold_validate'], 5)
            for name, parameter in model.state_dict().items():
                self.assertTrue(torch.equal(parameter, loaded.state_dict()[name]))


if __name__ == '__main__':
    unittest.main()
