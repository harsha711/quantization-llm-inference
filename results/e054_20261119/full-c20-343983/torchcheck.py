#!/usr/bin/env python3
"""E054 probe: the runtime's torch sees this GPU and runs an FP16 matmul (a script file: ARC prun flattens `python -c`)."""
import torch
a = torch.randn(256, 256, device='cuda').half()
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.version.cuda, float((a @ a).float().abs().sum()), flush=True)
