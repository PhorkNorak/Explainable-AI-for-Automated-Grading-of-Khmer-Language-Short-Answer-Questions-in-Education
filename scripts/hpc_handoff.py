"""Create an evidence-isolated snapshot, then run explicit HPC stages there.

No training is performed by prepare. Never copy existing result directories.
"""

# Support direct CLI execution from a source checkout.
import os as _layout_os
import sys as _layout_sys
_layout_root = _layout_os.path.dirname(_layout_os.path.dirname(_layout_os.path.abspath(__file__)))
for _layout_path in (_layout_root, _layout_os.path.join(_layout_root, "src")):
    if _layout_path not in _layout_sys.path:
        _layout_sys.path.insert(0, _layout_path)
from khmer_asag.paths import project_path

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    'classical': [['experiments/exp01_tfidf_baseline.py', '--datasets', 'no10c']],
    'bilstm': [['experiments/exp05_bilstm.py', '--datasets', 'no10c']],
    'encoder': [['experiments/exp06_transformer.py', '--datasets', 'full', 'no10c'],
                ['experiments/exp03b_maxfeat_neural.py', '--datasets', 'full', 'no10c']],
    'llm': [['experiments/exp08_llm_finetune.py', '--models', 'both', '--datasets', 'no10c',
             '--input', fmt, '--resume'] + extra
            for fmt in ('qar', 'ra') for extra in (['--zeroshot'], ['--epochs', '10'])],
    'xai': [['experiments/exp09_xai.py', '--families', 'classical', 'bilstm', 'encoder', 'llm',
             '--dataset', 'no10c', '--sample', '80', '--shap-max-evals', '2048',
             '--output-dir', 'local/results/results_xai/audit_compatible_samples']],
}

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def preflight_data():
    """Refuse out-of-scale ground truth before launching any training process."""
    for name in ('dataset.csv', 'dataset_no_10c_biology.csv'):
        path = ROOT/'local/data'/name
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for line, raw in enumerate(csv.DictReader(stream), 2):
                row = {key.strip(): value for key, value in raw.items()}
                if any(not row.get(key, '').strip() for key in ('Question', 'Reference', 'Answer')):
                    continue  # Mirrors the documented incomplete-text exclusion.
                try:
                    score, maximum = float(row['Student Score']), float(row['Max Score'])
                    valid = (math.isfinite(score) and math.isfinite(maximum)
                             and maximum > 0 and 0 <= score <= maximum
                             and score.is_integer() and maximum.is_integer())
                except (KeyError, ValueError, TypeError):
                    valid = False
                if not valid:
                    raise ValueError(f'{path}:{line}: invalid awarded score/maximum; '
                                     'confirm the original record before preparing a corrected dataset')

def expected_runs(stage):
    """Paths required by the explicit grid commands above, without importing ML."""
    for prep in ('raw', 'clean', 'segment'):
        for fmt in ('ra', 'qar'):
            if stage == 'classical':
                for model in ('tfidf_cos', 'tfidf_svr'):
                    yield ROOT/'local/results'/f'results_no10c/runs/{prep}_{fmt}_{model}'
            elif stage == 'bilstm':
                yield ROOT/'local/results'/f'results_no10c_v05_bilstm/runs/{prep}_{fmt}_bilstm'
            elif stage == 'encoder':
                for dataset in ('full', 'no10c'):
                    yield ROOT/'local/results'/f'results_{dataset}_v03b_maxfeat_neural/runs/{prep}_{fmt}_bilstm_maxfeat'
                    for arch in ('dual', 'cross'):
                        for model in ('mbert', 'xlmr', 'gte'):
                            yield ROOT/'local/results'/f'results_{dataset}_v06_transformer/runs/{prep}_{fmt}_{arch}_{model}'
                            yield ROOT/'local/results'/f'results_{dataset}_v03b_maxfeat_neural/runs/{prep}_{fmt}_{arch}_{model}_maxfeat'
    if stage == 'llm':
        for model in ('gemma4_e4b', 'qwen35_4b', 'sealion_v45_e2b'):
            for fmt in ('ra', 'qar'):
                yield ROOT/'local/results'/f'results_no10c_v08_llm_{model}/runs/clean_{fmt}_{model}'
                yield ROOT/'local/results'/f'results_no10c_v08z_llm_{model}_zeroshot/runs/zeroshot_{fmt}_{model}'

def validate_outputs(stage):
    if stage == 'xai':
        folder = ROOT/'local/results/results_xai/audit_compatible_samples'
        leaderboard = folder/'faithfulness_leaderboard.csv'
        if not leaderboard.exists():
            raise RuntimeError('XAI leaderboard was not produced')
        with leaderboard.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        found = {r['family'] for r in rows if r['explainer'] == 'shap' and int(r['n_explained']) > 0}
        if found != {'classical', 'bilstm', 'encoder', 'llm'}:
            raise RuntimeError(f'Incomplete XAI families: {sorted(found)}')
        for family in found:
            if not (folder/f'{family}_sample_manifest.json').is_file():
                raise RuntimeError(f'Missing sample manifest: {family}')
            if not list((folder/'attributions'/family/'shap').glob('*.json')):
                raise RuntimeError(f'Missing per-answer attributions: {family}')
    else:
        missing = [str(folder/name) for folder in expected_runs(stage)
                   for name in ('config.json', 'metrics.json', 'predictions_test.csv', 'predictions_val.csv')
                   if not (folder/name).is_file()]
        if missing:
            raise RuntimeError(f'{len(missing)} required stage artifacts missing; first: {missing[0]}')

def prepare(dest):
    dest = dest.resolve()
    if dest == ROOT or ROOT in dest.parents:
        raise ValueError('Snapshot must be outside the source repository')
    if dest.exists():
        raise FileExistsError('Use a new directory to preserve evidence')
    candidates = list(ROOT.glob('requirements*.txt')) + [ROOT/'pyproject.toml']
    for directory in ('src', 'experiments', 'scripts', 'tests'):
        candidates.extend(p for p in (ROOT/directory).rglob('*.py') if '__pycache__' not in p.parts)
    candidates += [ROOT/'local/data/dataset.csv', ROOT/'local/data/dataset_no_10c_biology.csv']
    manifest = {}
    for src in candidates:
        relative = src.relative_to(ROOT)
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        manifest[relative.as_posix()] = sha(target)
    (dest/'local/audit_inputs.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Prepared {dest}; no models trained and no existing evidence copied')

def run(stage):
    manifest_path = ROOT/'local/audit_inputs.json'
    if not manifest_path.exists():
        raise RuntimeError('Run stages only from a prepared snapshot')
    for rel, expected in json.loads(manifest_path.read_text()).items():
        if sha(ROOT/rel) != expected:
            raise RuntimeError(f'Input changed since snapshot: {rel}')
    preflight_data()
    records = ROOT/'local/logs/audit_execution'
    records.mkdir(parents=True, exist_ok=True)
    receipt = records/f'{stage}.json'
    if receipt.exists():
        raise FileExistsError(f'Stage receipt already exists: {receipt}; use a fresh snapshot')
    completed = []
    for number, command in enumerate(STAGES[stage]):
        log = records/f'{stage}_{number}.log'
        if log.exists():
            raise FileExistsError(f'Existing attempt preserved: {log}; use a fresh snapshot')
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.run([sys.executable, '-u', *command], cwd=ROOT,
                                     stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode or 'FAILED' in log.read_text(encoding='utf-8', errors='replace'):
            raise RuntimeError(f'Stage failed; inspect {log}')
        completed.append(command)
    validate_outputs(stage)
    # Artifact hashes link outputs to this immutable source/data snapshot.
    artifacts = {p.relative_to(ROOT).as_posix(): sha(p)
                 for p in (ROOT/'local/results').rglob('*') if p.is_file()}
    if not artifacts:
        raise RuntimeError('No result artifacts produced')
    with (records/f'{stage}_environment.txt').open('w', encoding='utf-8') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    receipt.write_text(json.dumps({'commands': completed, 'input_manifest_sha256': sha(manifest_path),
                                  'artifacts': artifacts,
                                  'status': 'execution_recorded_validation_review_required'}, indent=2), encoding='utf-8')
    print(f'Execution recorded in {receipt}; metric and selection review still required')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', *STAGES])
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        if args.destination is None:
            parser.error('prepare requires --destination')
        prepare(args.destination)
    else:
        run(args.action)
