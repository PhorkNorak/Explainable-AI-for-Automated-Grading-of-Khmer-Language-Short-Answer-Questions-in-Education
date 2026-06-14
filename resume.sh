mkdir -p logs && (
  python -u experiments/exp06_transformer.py --resume &&
  python -u experiments/exp03b_maxfeat_neural.py &&
  python -u experiments/exp02_threshold_calibration.py --source v05_bilstm &&
  python -u experiments/exp02_threshold_calibration.py --source v06_transformer &&
  python -u experiments/exp02_threshold_calibration.py --source v03b_maxfeat_neural &&
  python -u experiments/exp07_ensemble.py &&
  python -u experiments/exp08_llm_finetune.py --models qwen35_4b --epochs 7 &&
  python -u experiments/exp09_xai.py --families classical bilstm encoder llm --dataset no10c_no0 &&
  python -u experiments/exp10_significance.py &&
  python -u experiments/exp11_cleaning_ablation.py &&
  python -u experiments/exp12_hparam_tuning.py &&
  python -u experiments/compare_all.py --topk 15
) 2>&1 | tee "logs/resume_$(date +%Y%m%d_%H%M%S).log"
