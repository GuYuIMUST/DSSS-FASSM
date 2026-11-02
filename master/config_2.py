"""DSSS search configuration used for the published paper.

The search runner uses the shared configuration schema in ``config.py``.  The
paper reports a search base learning rate of 2e-4, so that value is made
explicit here without duplicating the full configuration tree.
"""

from config import get_config as _get_config


def get_config(args):
    """Return the DSSS-search configuration used in the published paper."""
    config = _get_config(args)
    config.defrost()
    config.TRAIN.BASE_LR = 2e-4
    config.TRAIN.MIN_LR = 5e-6
    config.freeze()
    return config
