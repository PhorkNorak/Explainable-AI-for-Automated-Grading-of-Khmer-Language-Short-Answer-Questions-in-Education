# New experiments on HPC

The final report records completed experiments. This guide describes a separate future rerun, not work performed during documentation cleanup. Encoder and LLM training require HPC/GPU resources; no training or API calls are part of this cleanup.

## Before running

The author's edited local CSVs correct one historical score from 19 to 9, with the maximum remaining 15. Dataset version 1.0.1 records this correction; see the [dataset card](dataset.md). Preserve historical data and results. Record the corrected data hashes and new split membership for a new run: changing the derived label can change the label-stratified split even with seed 42. The source helper continues to reject invalid marks before launching training.

Download the public CSVs from the linked dataset hosts, or use approved local research copies, with compatible dependencies from `requirements.txt` and the required GPU environment and model access. Record source and data hashes, split identities, upstream model revisions, package versions and hardware. Git excludes the CSVs, old prediction files and model weights; dataset copies are distributed separately under CC BY-NC 4.0.

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
