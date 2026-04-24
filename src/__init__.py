"""Knee X-ray pipeline modules (numeric prefix = run order).

1_run_logging -> 2_preprocess -> 3_labels -> 4_manifest -> 5_dataset -> 6_model -> 7_train -> 8_eval_run -> 9_eval_report -> 10_compare (optional multi-run charts)

Python cannot ``from src.2_foo import ...``; use ``importlib.import_module("src.2_preprocess")`` or load via ``main.py``.
"""
