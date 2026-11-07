#!/usr/bin/env python3
"""Tiny calibration for launch-dump metadata and --error-exitcode behavior."""
import torch

value = torch.empty(1,device='cuda')
value.fill_(3.)
torch.cuda.synchronize()
assert value.item() == 3.
print('PASS: one harmless fill kernel',flush=True)
