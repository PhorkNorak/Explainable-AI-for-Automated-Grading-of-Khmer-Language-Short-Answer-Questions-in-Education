# References

Citations used in the slide deck and thesis. Every URL below was retrieved during literature
research for this project. **Verify each entry against the publisher's official record (DOI,
venue, year, author order) before final thesis submission** — the annotations are working
summaries, not copy-ready bibliography entries.

## ASAG — classical foundations & shared tasks

1. **Mohler, M., & Mihalcea, R. (2009).** Text-to-text semantic similarity for automatic short
   answer grading. *EACL 2009.* — Foundational similarity-based ASAG.
2. **Mohler, M., Bunescu, R., & Mihalcea, R. (2011).** Learning to grade short answer questions
   using semantic similarity measures and dependency graph alignments. *ACL 2011.* —
   Similarity + dependency-graph features; widely-used CS short-answer dataset.
3. **Dzikovska, M. O., et al. (2013).** SemEval-2013 Task 7: The Joint Student Response Analysis
   and 8th Recognizing Textual Entailment Challenge. *SEM 2013.*
   https://aclanthology.org/S13-2045/ — SciEntsBank / Beetle benchmark.

## ASAG — deep learning & transformers

4. **Sung, C., et al. (2022).** Survey on Automated Short Answer Grading with Deep Learning:
   from Word Embeddings to Transformers. *arXiv:2204.03503.*
   https://arxiv.org/abs/2204.03503 — survey structuring embeddings → sequential → attention.
5. **(AIED 2020).** Investigating Transformers for Automatic Short Answer Grading.
   https://link.springer.com/chapter/10.1007/978-3-030-52240-7_8 —
   PMC mirror: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7334688/
6. **(2023).** Improving the performance of automatic short answer grading using transfer
   learning and augmentation. *Engineering Applications of AI.*
   https://www.sciencedirect.com/science/article/abs/pii/S0952197623004761 —
   augmentation for low-resource ASAG.

## ASAG — LLM era

7. **Chang, L.-H., & Ginter, F. (2024).** Performance of the Pre-Trained Large Language Model
   GPT-4 on Automated Short Answer Grading. *arXiv:2309.09338.*
   https://arxiv.org/abs/2309.09338 — GPT-4 on SciEntsBank/Beetle; competitive with
   hand-engineered models, below specialized fine-tuned models.
8. **(LAK 2025).** Automatic Short Answer Grading in the LLM Era: Does GPT-4 with Prompt
   Engineering beat Traditional Models? https://dl.acm.org/doi/10.1145/3706468.3706481
   *(paywalled; cited for the qualitative finding only — confirm exact QWK figures from the
   PDF before quoting them.)*
9. **(IJAIED 2025).** Cross-prompt Pre-finetuning of Language Models for Short Answer Scoring.
   https://link.springer.com/article/10.1007/s40593-025-00474-w — QLoRA on linear layers;
   QWK as the SAS standard metric.
10. **(2025).** FusionASAG: An LLM-Enhanced Automatic Short Answer Grading Model.
    https://link.springer.com/chapter/10.1007/978-981-96-3735-5_4 — reports avg QWK ≈ 0.797.

## Arabic ASAG — closest comparable study (primary anchor)

11. **Soulimani, Y. A., El Achaak, L., & Bouhorma, M. (2024).** Deep learning based Arabic short
    answer grading in serious games. *International Journal of Electrical and Computer
    Engineering (IJECE).* https://ijece.iaescore.com/index.php/IJECE/article/view/31253
    — Self-collected Arabic dataset: **1,276 answers, 18 questions, 3 classes (0–2)**, 6th
    grade. Transformer: **train 95.67% / test 77.22% accuracy**; **test Cohen κ (unweighted) =
    0.5996**; paper notes the transformer overfits faster as epochs increase.
    *(Numbers verified directly from the PDF in `docs/reference_papers/`.)*
12. **Ouahrani, L., & Bennouar, D.** AR-ASAG: An Arabic Dataset for Automatic Short Answer
    Grading Evaluation. https://www.semanticscholar.org/paper/c634d94c47c1b9d761d0a5dd451cabec780e2831

## Khmer NLP / low-resource resources

13. **Buoy, R., et al. (2022).** Pretrained Models and Evaluation Data for the Khmer Language.
    *Tsinghua Science and Technology.* https://www.sciopen.com/article/10.26599/TST.2021.9010060
14. **(ACM TALLIP 2021).** Towards Tokenization and Part-of-Speech Tagging for Khmer: Data and
    Discussion. https://dl.acm.org/doi/fullHtml/10.1145/3464378
15. **Nom, V., et al. (2024).** KhmerST: A Low-Resource Khmer Scene Text Detection and
    Recognition Benchmark. *ACCV 2024.*
    https://openaccess.thecvf.com/content/ACCV2024/papers/Nom_KhmerST_A_Low-Resource_Khmer_Scene_Text_Detection_and_Recognition_Benchmark_ACCV_2024_paper.pdf
16. **Hoang, V. (2020).** khmer-nltk: Khmer language processing toolkit (word segmentation).
    https://github.com/VietHoang1512/khmer-nltk
17. **awesome-khmer-language** — curated collection of Khmer language resources.
    https://github.com/seanghay/awesome-khmer-language

## Explainable AI — methods & evaluation

20. **DeYoung, J., et al. (2020).** ERASER: A Benchmark to Evaluate Rationalized NLP Models.
    *ACL 2020.* arXiv:1911.03429 — defines **comprehensiveness** and **sufficiency**, the
    faithfulness metrics used here.
21. **Jain, S., & Wallace, B. C. (2019).** Attention is not Explanation. *NAACL 2019.*
    arXiv:1902.10186 — why raw attention weights are not necessarily faithful explanations.
22. **Wiegreffe, S., & Pinter, Y. (2019).** Attention is not not Explanation. *EMNLP 2019.*
    arXiv:1908.04626 — the counter-position; motivates *evaluating* attention via faithfulness
    rather than trusting or dismissing it.
23. **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** "Why Should I Trust You?": Explaining
    the Predictions of Any Classifier (**LIME**). *KDD 2016.* arXiv:1602.04938 — model-agnostic
    local explanation (occlusion in this thesis is a dependency-free relative).
24. **Lundberg, S. M., & Lee, S.-I. (2017).** A Unified Approach to Interpreting Model
    Predictions (**SHAP**). *NeurIPS 2017.* arXiv:1705.07874.
25. **Simonyan, K., Vedaldi, A., & Zisserman, A. (2014).** Deep Inside Convolutional Networks:
    Visualising Image Classification Models and Saliency Maps. *ICLR Workshop 2014.*
    arXiv:1312.6034 — origin of gradient-based saliency (used for the encoder pillar).
26. **Li, J., Chen, X., Hovy, E., & Jurafsky, D. (2016).** Visualizing and Understanding Neural
    Models in NLP. *NAACL 2016.* arXiv:1506.01066 — gradient×input attribution for text.

## Evaluation methodology

27. **Berg-Kirkpatrick, T., Burkett, D., & Klein, D. (2012).** An Empirical Investigation of
    Statistical Significance in NLP. *EMNLP 2012.* — motivates bootstrap/paired significance
    testing for the QWK/accuracy comparisons.
28. **Unseen-question / grouped evaluation** — cf. SemEval-2013 Task 7 "unseen domains/questions"
    (Dzikovska et al., ref. 3); our question-held-out split uses `sklearn` `GroupShuffleSplit`
    over `QuestionID`.

## Tools

18. **Unsloth** — efficient QLoRA fine-tuning library. https://github.com/unslothai/unsloth
19. **GTE-multilingual-base (Alibaba-NLP)** — multilingual text encoder used for the encoder
    pillar. https://huggingface.co/Alibaba-NLP/gte-multilingual-base

---

### Note on metric comparability
QWK is **dataset-dependent** (class count, scale, rater noise). Cross-paper QWK/κ values in
this thesis are **contextual positioning, not a fair benchmark**. Alaoui et al. report
**unweighted** Cohen's κ on a 3-class scale; this is not directly comparable to our 5-class
QWK. The defensible cross-study comparison is the **train/test generalization gap** on
accuracy (slide 5.6).
