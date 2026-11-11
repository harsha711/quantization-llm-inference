#!/usr/bin/env python3
"""E029: E026's att.py with the Marlin operator's prefill row limit raised from 2048 to 16384.

The limit is an assertion bound in the E017 operator wrapper; its workspace size does not depend on it above 64 rows.
Decode (one row) is unaffected. Arguments are att.py's.
"""
import importlib.util
import sys
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location('att', Path(__file__).resolve().parent / 'att.py')
att = importlib.util.module_from_spec(spec)
spec.loader.exec_module(att)
att.I.Operator.__init__.__defaults__ = (16384,)
with torch.inference_mode():
    target = Path(sys.argv[2])
    if sys.argv[1] == 'validate':
        target.mkdir(parents=True, exist_ok=False)
        att.validate(target, [int(c) for c in sys.argv[3:]])
    else:
        assert sys.argv[1] == 'time' and sys.argv[3] in att.SAVED
        att.timed(target, sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6]), Path(sys.argv[7]))
