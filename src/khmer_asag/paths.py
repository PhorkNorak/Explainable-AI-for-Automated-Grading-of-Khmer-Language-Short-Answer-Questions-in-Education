"""Repository source paths and private research-artifact locations.

Historical run-directory names are preserved inside local/results. Relative legacy
paths are resolved here so saved evidence need not be rewritten after relocation.
"""
from pathlib import Path
import os

PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
LOCAL_ROOT = REPOSITORY_ROOT / 'local'
DATA_DIR = LOCAL_ROOT / 'data'
RESULTS_ROOT = LOCAL_ROOT / 'results'
CHECKPOINTS_ROOT = LOCAL_ROOT / 'checkpoints'
LOGS_DIR = LOCAL_ROOT / 'logs'


def project_path(*parts):
    """Resolve an old project-relative path, preserving explicit absolute paths."""
    joined = os.path.join(*(os.fspath(p) for p in parts)) if parts else ''
    path = Path(joined)
    if path.is_absolute():
        return str(path)
    if not path.parts:
        return str(REPOSITORY_ROOT)
    head, *tail = path.parts
    if head == 'data':
        base = DATA_DIR
    elif head == 'results':
        base = CHECKPOINTS_ROOT / 'results'
    elif head.startswith(('results_', 'xai_visuals_')):
        base = RESULTS_ROOT / head
    elif head == 'logs':
        base = LOGS_DIR
    elif head in {'publish', 'output', 'tmp', '.release', 'outputs'}:
        base = LOCAL_ROOT / 'builds' / head
    elif head in {'distribution', 'kaggle-check'}:
        base = LOCAL_ROOT / 'archive' / head
    elif head == '.hfcache':
        base = LOCAL_ROOT / 'cache' / head
    elif head in {'models', 'xai'} or head in {'config.py','data.py','preprocess.py','evaluate.py','train.py'}:
        base = PACKAGE_ROOT / head
    else:
        base = REPOSITORY_ROOT / head
    return str(base.joinpath(*tail))
