#!/usr/bin/env python3
"""Quick test: can LUT4 lm_head models load on ANE?"""
import coremltools as ct
import time, signal, sys

def handler(signum, frame):
    print(f'TIMEOUT after 90s — model loading hung', flush=True)
    sys.exit(1)

signal.signal(signal.SIGALRM, handler)
signal.alarm(90)

gs = int(sys.argv[1]) if len(sys.argv) > 1 else 1
path = f'/tmp/lmhead_groupsize_sweep/lm_head_LUT4_gs{gs}.mlpackage'
print(f'Loading gs={gs} with CPU_AND_NE...', flush=True)
t0 = time.time()
try:
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f'Loaded in {time.time()-t0:.1f}s', flush=True)
    del m
except Exception as e:
    print(f'Error after {time.time()-t0:.1f}s: {e}', flush=True)
