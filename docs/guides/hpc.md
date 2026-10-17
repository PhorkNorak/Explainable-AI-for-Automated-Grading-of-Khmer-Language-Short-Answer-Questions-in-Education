# New experiments on HPC

The final report records completed experiments. This guide describes a separate future rerun, not work performed during documentation cleanup. Encoder and LLM training require HPC/GPU resources; no training or API calls are part of this cleanup.

## Before running

A retained source record has a score of 19 with a maximum of 15. Confirm the original awarded mark and maximum before using the affected data. Do not silently clip, delete or guess a correction. Preserve the original data and prepare any approved correction in a separate private checkout with its change record and hashes. The source helper rejects invalid marks before launching training.

Use approved private data and compatible dependencies from `requirements.txt`, plus the required GPU environment and model access. Record source and data hashes, split identities, upstream model revisions, package versions and hardware. The public repository does not supply student data, old prediction files or model weights.

## Isolated workflow

From a prepared checkout with approved, valid input data, choose a new absolute scratch path outside the checkout:

```bash
python scripts/hpc_handoff.py prepare --destination /approved/scratch/khmer-run01
cd /approved/scratch/khmer-run01
python scripts/hpc_handoff.py classical
python scripts/hpc_handoff.py bilstm
python scripts/hpc_handoff.py encoder
python scripts/hpc_handoff.py llm
python scripts/hpc_handoff.py xai
```

The helper copies source and approved data into a new snapshot, not legacy results. It refuses to reuse attempted stages and keeps failed logs. Review a failure and prepare a fresh snapshot rather than deleting evidence to force a retry. A completed receipt records output existence and hashes; it does not prove that an experiment is scientifically verified.

Select models using validation results, verify checkpoint reload and predictions, and keep full-set encoder results separate from comparisons on the shared no10c test rows. Preserve grade 0 and distinguish calibrated labels from uncalibrated rounding. Compare each adapted LLM with its own base revision and prompt. SHAP refits must not be described as reloaded champion explanations without compatible checkpoint provenance.

After reviewing compatible selections and the explicit champion mapping, write new summary results to a fresh directory:

```bash
python experiments/exp10_significance.py --output-dir local/results/results_stats/reviewed_run01
```

Keep the report's recorded findings separate from any new results. See [evidence limitations](report_evidence.md), the [experiment registry](../../experiments/README.md) and the local historical audit records for further context.
