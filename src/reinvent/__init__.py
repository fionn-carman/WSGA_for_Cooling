# REINVENT — Reinforcement Learning for de novo molecular design
# Standard REINVENT (Olivecrona et al. 2017) with augmented likelihood.

import os as _os

# Prevent OMP/MKL segfault when PyTorch + XGBoost coexist in same process
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
