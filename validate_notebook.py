"""Validate ieeecis_exploration.ipynb has all required cells and code."""
import json
import sys

PASS = '\033[92mPASS\033[0m'
FAIL = '\033[91mFAIL\033[0m'
errors = []

def check(condition, msg):
    if condition:
        print(f'  {PASS} {msg}')
    else:
        print(f'  {FAIL} {msg}')
        errors.append(msg)

def get_all_code(nb):
    """Concatenate all code cell sources."""
    return '\n'.join(
        ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        for c in nb['cells'] if c['cell_type'] == 'code'
    )

def find_cell_with(nb, text):
    """Find index of first code cell containing text."""
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] == 'code':
            src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
            if text in src:
                return i
    return -1

# Load notebook
print('Loading notebook...')
with open('/Users/camiloavila/Documents/fraudproject/ieeecis_exploration.ipynb') as f:
    nb = json.load(f)

code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
all_code = get_all_code(nb)

print(f'\n=== Structure ===')
check(len(nb['cells']) >= 60, f'At least 60 cells (found {len(nb["cells"])})')
check(len(code_cells) >= 30, f'At least 30 code cells (found {len(code_cells)})')
check(nb['nbformat'] == 4, 'nbformat is 4')

# 1. Required imports
print(f'\n=== Imports ===')
required_imports = [
    ('import numpy', 'numpy'),
    ('import pandas', 'pandas'),
    ('import matplotlib', 'matplotlib'),
    ('import seaborn', 'seaborn'),
    ('from fdb.datasets import FraudDatasetBenchmark', 'FDB'),
    ('from sklearn.ensemble import RandomForestClassifier', 'sklearn RandomForest'),
    ('from sklearn.metrics import', 'sklearn metrics'),
    ('import xgboost', 'xgboost'),
    ('import optuna', 'optuna'),
    ('from sklearn.ensemble import StackingClassifier', 'StackingClassifier'),
    ('from scipy.stats import ks_2samp', 'scipy ks_2samp'),
    ('from sklearn.semi_supervised import SelfTrainingClassifier', 'SelfTrainingClassifier'),
    ('import shap', 'shap'),
]
for import_str, name in required_imports:
    check(import_str in all_code, f'Import: {name}')

# 2. Data loading
print(f'\n=== Data Loading ===')
check('FraudDatasetBenchmark' in all_code, 'FraudDatasetBenchmark used')
check("key='ieeecis'" in all_code or 'key="ieeecis"' in all_code, 'ieeecis dataset key')
check('load_pre_downloaded=True' in all_code, 'load_pre_downloaded=True')

# 3. Model training cells
print(f'\n=== Model Training Cells ===')

# RandomForest
check('rf_baseline = RandomForestClassifier(' in all_code, 'RF Baseline training cell')
check('rf_balanced = RandomForestClassifier(' in all_code, 'RF Balanced training cell')
check("class_weight='balanced'" in all_code, 'class_weight balanced')

# XGBoost
check('xgb_model = xgb.XGBClassifier(' in all_code, 'XGBoost base training cell')
check('scale_pos_weight' in all_code, 'scale_pos_weight defined')

# Feature Engineering + Temporal
check('def add_features(' in all_code, 'add_features function defined')
check('hour_of_day' in all_code, 'hour_of_day temporal feature')
check('day_of_week' in all_code, 'day_of_week temporal feature')
check('is_weekend' in all_code, 'is_weekend temporal feature')
check('xgb_eng = xgb.XGBClassifier(' in all_code, 'XGBoost + FE training cell')

# Optuna
check('optuna.create_study(' in all_code, 'Optuna study created')
check('study.optimize(' in all_code, 'Optuna optimize called')
check('xgb_tuned = xgb.XGBClassifier(' in all_code, 'XGBoost Tuned training cell')

# Stacking
check('stacking_clf = StackingClassifier(' in all_code, 'StackingClassifier created')
check("final_estimator=meta_learner" in all_code, 'meta_learner in stacking')
check('meta_learner = xgb.XGBClassifier(' in all_code, 'XGBoost meta-learner')

# Semi-supervised
check('semi_model = SelfTrainingClassifier(' in all_code, 'SelfTrainingClassifier cell exists')
check('estimator=base_xgb' in all_code, 'estimator= (not base_estimator=)')
check("criterion='threshold'" in all_code, "criterion='threshold' (not 'thresholding')")
check('semi_model.fit(' in all_code, 'semi_model.fit() called')
check('semi_metrics = print_metrics(' in all_code, 'semi_metrics assigned')
check('y_train_semi' in all_code, 'y_train_semi labels created')

# Concept Drift
check("train_drift['month']" in all_code or "train_drift['month']" in all_code, 'Concept drift month column')
check('drift_results' in all_code, 'drift_results dict')
check('ks_2samp' in all_code, 'KS test called')

# 4. Metrics and evaluation
print(f'\n=== Metrics & Evaluation ===')
check('def print_metrics(' in all_code, 'print_metrics function defined')
check('def get_tpr_at_fpr(' in all_code, 'get_tpr_at_fpr function defined')
check('roc_auc_score' in all_code, 'roc_auc_score used')
check('precision_recall_curve' in all_code, 'precision_recall_curve used')
check('confusion_matrix' in all_code, 'confusion_matrix used')
check('classification_report' in all_code, 'classification_report used')

# 5. SHAP
print(f'\n=== SHAP ===')
check('shap.TreeExplainer(' in all_code, 'SHAP TreeExplainer')
check('shap.summary_plot(' in all_code, 'SHAP summary_plot')

# 6. Comparison table
print(f'\n=== Comparison Table ===')
check("'RF Baseline'" in all_code, 'RF Baseline in comparison')
check("'RF Balanced'" in all_code, 'RF Balanced in comparison')
check("'XGBoost Tuned'" in all_code, 'XGBoost Tuned in comparison')
check("'Stacked Ensemble'" in all_code, 'Stacked Ensemble in comparison')

# 7. Cell order checks
print(f'\n=== Cell Order ===')
idx_import_selftraining = find_cell_with(nb, 'from sklearn.semi_supervised import SelfTrainingClassifier')
idx_dataprep = find_cell_with(nb, 'y_train_semi = y_train.copy()')
idx_training = find_cell_with(nb, 'semi_model = SelfTrainingClassifier(')
idx_comparison = find_cell_with(nb, 'supervised_full = xgb_eng_metrics')

check(idx_import_selftraining >= 0, f'Import cell found (cell {idx_import_selftraining})')
check(idx_dataprep >= 0, f'Data prep cell found (cell {idx_dataprep})')
check(idx_training >= 0, f'Training cell found (cell {idx_training})')
check(idx_comparison >= 0, f'Comparison cell found (cell {idx_comparison})')

if idx_import_selftraining >= 0 and idx_dataprep >= 0 and idx_training >= 0 and idx_comparison >= 0:
    check(idx_import_selftraining <= idx_dataprep < idx_training < idx_comparison,
          f'Order correct: import({idx_import_selftraining}) <= prep({idx_dataprep}) < train({idx_training}) < compare({idx_comparison})')

# 8. No undefined references
print(f'\n=== Variable References ===')
assigned_vars = set()
referenced_before_assignment = []

for c in nb['cells']:
    if c['cell_type'] == 'code':
        src = ''.join(c['source']) if isinstance(c['source'], list) else c['source']
        # Check if key vars are assigned in this cell
        for var in ['semi_metrics', 'xgb_tuned_metrics', 'stack_metrics', 'rf_metrics', 'xgb_metrics',
                     'xgb_eng_metrics', 'rf_bal_metrics', 'best_params', 'scale_pos_weight']:
            if f'{var} =' in src or f'{var}=' in src:
                assigned_vars.add(var)
        # Check references
        for var in ['semi_metrics', 'xgb_tuned_metrics', 'stack_metrics']:
            if var in src and f'{var} =' not in src and f'{var}=' not in src:
                if var not in assigned_vars:
                    referenced_before_assignment.append((var, src.split('\n')[0][:60]))

check(len(referenced_before_assignment) == 0,
      f'No undefined variable references' + (f' (issues: {referenced_before_assignment})' if referenced_before_assignment else ''))

# Summary
print(f'\n{"="*50}')
if errors:
    print(f'\033[91m{len(errors)} check(s) FAILED:\033[0m')
    for e in errors:
        print(f'  - {e}')
    sys.exit(1)
else:
    print(f'\033[92mAll checks PASSED\033[0m')
    sys.exit(0)
