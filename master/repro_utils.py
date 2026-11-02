"""Small reproducibility helpers shared by search, training, and evaluation."""

import random

import numpy as np
import torch


def seed_everything(seed):
    """Seed every RNG used by the data transforms and PyTorch loaders."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cycling_batches(loader):
    """Yield every batch in order and restart only after exhaustion."""
    if len(loader) == 0:
        raise ValueError('cannot cycle an empty data loader')
    while True:
        for batch in loader:
            yield batch


def capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        'python': random.getstate(),
        # Store only primitive containers so the checkpoint can be opened with
        # torch.load(weights_only=True).
        'numpy': {
            'bit_generator': numpy_state[0],
            'state': numpy_state[1].tolist(),
            'position': numpy_state[2],
            'has_gauss': numpy_state[3],
            'cached_gaussian': numpy_state[4],
        },
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state['python'])
    numpy_state = state['numpy']
    if isinstance(numpy_state, dict):
        numpy_state = (
            numpy_state['bit_generator'],
            np.asarray(numpy_state['state'], dtype=np.uint32),
            numpy_state['position'],
            numpy_state['has_gauss'],
            numpy_state['cached_gaussian'],
        )
    np.random.set_state(numpy_state)
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
