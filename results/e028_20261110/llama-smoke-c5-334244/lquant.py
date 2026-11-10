#!/usr/bin/env python3
"""E028: llama.cpp's own Q4_K_M quantization of the FP16 GGUF. lquant.py IN OUT"""
import ctypes
import sys

import llama_cpp

llama_cpp.llama_backend_init()
params = llama_cpp.llama_model_quantize_default_params()
params.ftype, params.nthread = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M, 16
assert llama_cpp.llama_model_quantize(sys.argv[1].encode(), sys.argv[2].encode(), ctypes.byref(params)) == 0
