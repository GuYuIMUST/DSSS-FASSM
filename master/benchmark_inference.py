"""Measure FASSM model-only CUDA inference using the published protocol."""

import argparse
import json
import os

import numpy as np
import torch

from checkpointing import load_fassm_checkpoint
from repro_utils import seed_everything


def main():
    parser = argparse.ArgumentParser(description='Benchmark FASSM model-only CUDA inference')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--passes', type=int, default=200)
    parser.add_argument('--output', default='outputs/benchmark_inference.json')
    parser.add_argument('--allow_legacy_pickle', action='store_true')
    args = parser.parse_args()
    if not os.path.isfile(args.checkpoint):
        parser.error('--checkpoint must be an existing file')
    if min(args.warmup, args.rounds, args.passes) <= 0:
        parser.error('--warmup, --rounds, and --passes must all be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-capable GPU is required for inference benchmarking')

    seed_everything(args.seed)
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda', args.gpu)
    model, metadata = load_fassm_checkpoint(
        args.checkpoint, device,
        defaults={'init_channels': 20, 'layers': 20, 'arch': 'DARTS_MAMBA',
                  'auxiliary': True, 'num_classes': 2},
        allow_legacy_pickle=args.allow_legacy_pickle,
    )
    model.eval()
    inputs = torch.randn(1, 1, 32, 32, 32, device=device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(inputs)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        timings_ms = []
        for _ in range(args.rounds):
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.passes)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.passes)]
            for start, end in zip(starts, ends):
                start.record()
                model(inputs)
                end.record()
            torch.cuda.synchronize(device)
            timings_ms.extend(start.elapsed_time(end) for start, end in zip(starts, ends))

    timings_ms = np.asarray(timings_ms, dtype=np.float64)
    result = {
        'protocol': {
            'input_shape': [1, 1, 32, 32, 32],
            'batch_size': 1,
            'warmup_iterations': args.warmup,
            'measurement_rounds': args.rounds,
            'passes_per_round': args.passes,
            'scope': 'CUDA forward propagation only',
        },
        'mean_ms': float(np.mean(timings_ms)),
        'median_ms': float(np.median(timings_ms)),
        'p95_ms': float(np.percentile(timings_ms, 95)),
        'peak_allocated_mib': float(torch.cuda.max_memory_allocated(device) / 2 ** 20),
        'peak_reserved_mib': float(torch.cuda.max_memory_reserved(device) / 2 ** 20),
        'device': torch.cuda.get_device_name(device),
        'checkpoint': os.path.abspath(args.checkpoint),
        'checkpoint_sha256': metadata['sha256'],
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
