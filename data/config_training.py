"""Local paths for LUNA16 preprocessing.

Set the environment variables below before running ``prepare.py``.  The
repository deliberately contains no dataset-specific paths or image data.
``DSSS_LUNA_ROOT`` must contain the LUNA16 ``subset0`` through ``subset9``
directories.  ``DSSS_LUNA_SEGMENT`` is the lung-segmentation directory used by
the legacy preprocessing implementation.
"""

import os


def _env(name):
    return os.environ.get(name, '').strip()


_LUNA_ROOT = _env('DSSS_LUNA_ROOT')
_PREPROCESS_ROOT = _env('DSSS_PREPROCESS_ROOT')

# ``full_prep`` is retained for the legacy DeepLung-style preprocessing path.
# It is not the default entry point, but all of its settings are explicit here.
config = {
    'luna_data': _LUNA_ROOT,
    'luna_segment': _env('DSSS_LUNA_SEGMENT'),
    'luna_label': _env('DSSS_LUNA_LABEL'),
    'preprocess_result_path': _PREPROCESS_ROOT,
    'train_preprocess_result_path': _env('DSSS_TRAIN_PREPROCESS_ROOT'),
    'val_preprocess_result_path': _env('DSSS_VAL_PREPROCESS_ROOT'),
    'test_preprocess_result_path': _env('DSSS_TEST_PREPROCESS_ROOT'),
    'train_data_path': _env('DSSS_TRAIN_DATA_ROOT'),
    'val_data_path': _env('DSSS_VAL_DATA_ROOT'),
    'test_data_path': _env('DSSS_TEST_DATA_ROOT'),
    'train_annos_path': _env('DSSS_TRAIN_ANNOS_ROOT'),
    'val_annos_path': _env('DSSS_VAL_ANNOS_ROOT'),
    'test_annos_path': _env('DSSS_TEST_ANNOS_ROOT'),
    'black_list': [],
}
