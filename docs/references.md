# References (verified, with download links)

Every entry below corresponds to a key in `thesis/refs.bib` (= `paper/refs.bib`) and was
**web-verified during the reference audit** (title, authors, venue, year, and a working
DOI / arXiv / publisher link). Entries with a local copy are marked **[PDF in
`docs/reference_papers/`]**. Links go to the official/open record so each can be downloaded.

Still verify author order and page numbers against the publisher's record before final binding;
the audit confirmed each paper exists and is the correct work, not every bibliographic field.

> **Thesis vs. full list (recency-first 15-paper core).** The final thesis cites a lean **15-paper
> core** (its bibliography prints only those), chosen to be **recent (last ~5 years, 2021-2026)** wherever
> possible: `sung2022survey, chang2024gpt4, alaoui2024, tan2025review, dettmers2023qlora,
> pinto2025attribution, lyu2024faithfulsurvey, lachana2025dutch, xu2025qwenscore, buoy2022khmer,
> huot2025cambodia` (all 2022-2025), plus four slightly-older entries kept because the thesis directly
> *uses* them or they are uniquely relevant: `devlin2019bert` (mBERT, the encoder), `deyoung2020eraser`
> (ERASER, the faithfulness metric), `khmernltk2020` (the Khmer segmenter used), and `kumar2020explainable`
> (explainable AES, QWK + exact/adjacent + SHAP, the closest prior work).
>
> The remaining entries listed below are verified and available (cited by the companion paper or
> consulted during the review) but are **not** cited in the final thesis.

---

## ASAG: foundations and shared tasks

- **`mohler2011`** Mohler, Bunescu, Mihalcea (2011). *Learning to grade short answer questions using
  semantic similarity measures and dependency graph alignments.* ACL 2011.
  https://aclanthology.org/P11-1076/
- **`dzikovska2013`** Dzikovska et al. (2013). *SemEval-2013 Task 7: The Joint Student Response
  Analysis and 8th RTE Challenge.* *SEM 2013. https://aclanthology.org/S13-2045/

## ASAG: deep learning and reviews

- **`sung2022survey`** Sung, Dhamecha, Mukhi (2022). *A survey on automated short answer grading with
  deep learning: from word embeddings to transformers.* arXiv:2204.03503.
  https://arxiv.org/abs/2204.03503
- **`tan2025review`** Tan, Hu, Yeo, Cheong (2025). *A comprehensive review on automated grading
  systems in STEM using AI techniques.* Mathematics 13(17):2828.
  https://doi.org/10.3390/math13172828 **[PDF in `docs/reference_papers/`]**

## ASAG: LLM era

- **`chang2024gpt4`** Chang, Ginter (2024). *Performance of the pre-trained LLM GPT-4 on automated
  short answer grading.* arXiv:2309.09338. https://arxiv.org/abs/2309.09338
- **`sasbench2025`** Lai et al. (2025). *SAS-Bench: A fine-grained benchmark for evaluating short
  answer scoring with large language models.* arXiv:2505.07247. https://arxiv.org/abs/2505.07247
- **`graderag2025`** Chu et al. (2025). *Enhancing LLM-based short answer grading with
  retrieval-augmented generation.* arXiv:2504.05276. https://arxiv.org/abs/2504.05276
- **`neurosymbolic2024asag`** Künnecke, Filighera, Leong, Steuer (2024). *Enhancing multi-domain
  automatic short answer grading through an explainable neuro-symbolic pipeline.* arXiv:2403.01811.
  https://arxiv.org/abs/2403.01811

## Arabic ASAG (the closest comparable study / anchor)

- **`alaoui2024`** Soulimani (Alaoui), El Achaak, Bouhorma (2024). *Deep learning based Arabic short
  answer grading in serious games.* IJECE 14(1). https://ijece.iaescore.com/index.php/IJECE/article/view/31253
  **[PDF in `docs/reference_papers/`]** (1,276 answers, 18 questions, 3 classes; test acc 77.22%,
  unweighted kappa 0.60; verified directly from the PDF.)
- **`alqurashi2025arabic`** Alqurashi, Alharbi, Sabbeh (2025). *An automatic grading system for
  Arabic language short-answer questions using deep learning.* ETASR 15(5):26665-26675.
  https://etasr.com/index.php/ETASR/article/view/10917 **[PDF in `docs/reference_papers/`]**

## Transformers and pre-trained encoders

- **`vaswani2017attention`** Vaswani et al. (2017). *Attention is all you need.* NeurIPS.
  https://arxiv.org/abs/1706.03762
- **`devlin2019bert`** Devlin, Chang, Lee, Toutanova (2019). *BERT.* NAACL-HLT.
  https://aclanthology.org/N19-1423/
- **`conneau2020xlmr`** Conneau et al. (2020). *Unsupervised cross-lingual representation learning at
  scale (XLM-R).* ACL. https://aclanthology.org/2020.acl-main.747/
- **`reimers2019sbert`** Reimers, Gurevych (2019). *Sentence-BERT.* EMNLP-IJCNLP.
  https://aclanthology.org/D19-1410/
- **`bahdanau2015attention`** Bahdanau, Cho, Bengio (2015). *Neural machine translation by jointly
  learning to align and translate.* ICLR. https://arxiv.org/abs/1409.0473

## LLM adaptation (the LLM pillar)

- **`dettmers2023qlora`** Dettmers, Pagnoni, Holtzman, Zettlemoyer (2023). *QLoRA: Efficient
  finetuning of quantized LLMs.* NeurIPS. https://arxiv.org/abs/2305.14314
- **`hu2022lora`** Hu et al. (2022). *LoRA: Low-rank adaptation of large language models.* ICLR.
  https://arxiv.org/abs/2106.09685

## Explainable AI: attribution methods

- **`ribeiro2016lime`** Ribeiro, Singh, Guestrin (2016). *"Why should I trust you?" (LIME).* KDD.
  https://arxiv.org/abs/1602.04938
- **`lundberg2017shap`** Lundberg, Lee (2017). *A unified approach to interpreting model predictions
  (SHAP).* NeurIPS. https://arxiv.org/abs/1705.07874
- **`simonyan2014saliency`** Simonyan, Vedaldi, Zisserman (2014). *Deep inside convolutional networks:
  saliency maps.* ICLR Workshop. https://arxiv.org/abs/1312.6034
- **`li2016visualizing`** Li, Chen, Hovy, Jurafsky (2016). *Visualizing and understanding neural
  models in NLP.* NAACL-HLT. https://arxiv.org/abs/1506.01066
- **`sundararajan2017ig`** Sundararajan, Taly, Yan (2017). *Axiomatic attribution for deep networks
  (Integrated Gradients).* ICML. https://arxiv.org/abs/1703.01365

## Explainable AI: the attention debate and faithfulness

- **`jain2019attention`** Jain, Wallace (2019). *Attention is not Explanation.* NAACL.
  https://arxiv.org/abs/1902.10186
- **`wiegreffe2019attention`** Wiegreffe, Pinter (2019). *Attention is not not Explanation.* EMNLP.
  https://arxiv.org/abs/1908.04626
- **`jacovi2020faithfulness`** Jacovi, Goldberg (2020). *Towards faithfully interpretable NLP: how
  should we define and evaluate faithfulness?* ACL. https://arxiv.org/abs/2004.03685
- **`lyu2024faithfulsurvey`** Lyu, Apidianaki, Callison-Burch (2024). *Towards faithful model
  explanation in NLP: a survey.* Computational Linguistics 50(2).
  https://doi.org/10.1162/coli_a_00511
- **`deyoung2020eraser`** DeYoung et al. (2020). *ERASER: a benchmark to evaluate rationalized NLP
  models.* ACL. https://arxiv.org/abs/1911.03429
- **`samek2017aopc`** Samek et al. (2017). *Evaluating the visualization of what a deep neural network
  has learned (AOPC).* IEEE TNNLS 28(11). https://arxiv.org/abs/1509.06321
- **`naopc2024`** Edin et al. (2024). *Normalized AOPC: fixing misleading faithfulness metrics for
  feature attribution explainability.* arXiv:2408.08137. https://arxiv.org/abs/2408.08137

## Explainability in scoring and feedback

- **`kumar2020explainable`** Kumar, Boulanger (2020). *Explainable automated essay scoring: deep
  learning really has pedagogical value.* Frontiers in Education 5:572367.
  https://doi.org/10.3389/feduc.2020.572367 **[PDF in `docs/reference_papers/`]**
- **`exasag2023`** Törnqvist, Mahamud, Mendez Guzman, Farazouli (2023). *ExASAG: explainable framework
  for automatic short answer grading.* BEA @ ACL. https://aclanthology.org/2023.bea-1.29/
- **`pinto2025attribution`** Pinto Jr., Shin (2025). *Evaluating the consistency and reliability of
  attribution methods in ASAG systems.* Journal of Educational Measurement 62(2):248-281.
  https://doi.org/10.1111/jedm.12438 (compares LIME, IG, HEDGE, and Leave-One-Out)
- **`nam2024grading`** Condor, Pardos (2024). *Explainable automatic grading with neural additive
  models.* arXiv:2405.00489 (also AIED 2024). https://arxiv.org/abs/2405.00489
- **`xu2025qwenscore`** Xu et al. (2025). *Explainable AI for education: enhancing essay scoring via
  rubric-aligned chain-of-thought prompting.* Preprints.org.
  https://doi.org/10.20944/preprints202504.2338.v1 **[PDF in `docs/reference_papers/`]**
- **`wei2022cot`** Wei et al. (2022). *Chain-of-thought prompting elicits reasoning in LLMs.* NeurIPS.
  https://arxiv.org/abs/2201.11903

## Deployed / human-in-the-loop ASAG with feedback

- **`condor2024`** Condor (2024). *Automatic grading and feedback for students' short written
  responses.* Ph.D. dissertation, UC Berkeley. https://escholarship.org/uc/item/4f54k5tf
  **[PDF in `docs/reference_papers/`]**
- **`engsaf2024`** Aggarwal, Sil, Raman, Bhattacharyya (2024). *"I understand why I got this grade":
  automatic short answer grading with feedback.* arXiv:2407.12818. https://arxiv.org/abs/2407.12818
- **`chillgrader2025`** Raikote, Randl, Miliou, Lakes, Papapetrou (2026). *CHiL(L)Grader: calibrated
  human-in-the-loop short-answer grading.* arXiv:2603.11957. https://arxiv.org/abs/2603.11957

## Cross-lingual and low-resource scoring

- **`li2024crosslingual`** Li, He (2024). *Zero-shot cross-lingual automated essay scoring.*
  LREC-COLING. https://aclanthology.org/2024.lrec-main.1558/ **[PDF in `docs/reference_papers/`]**
- **`lachana2025dutch`** Lachana (2025). *Automated scoring systems for open-ended questions in Dutch
  education.* M.Sc. thesis, Utrecht University (with CITO).
  https://studenttheses.uu.nl/handle/20.500.12932/50896 **[PDF in `docs/reference_papers/`]**

## Khmer NLP and the Cambodian context

- **`buoy2022khmer`** Buoy et al. (2022). *Pretrained models and evaluation data for the Khmer
  language.* Tsinghua Science and Technology. https://doi.org/10.26599/TST.2021.9010060
- **`khmernltk2020`** Hoang (2020). *khmer-nltk: Khmer NLP toolkit (word segmentation).*
  https://github.com/VietHoang1512/khmer-nltk
- **`huot2025cambodia`** Huot et al. (2025). *Educating in the age of AI: preparing Cambodian teachers
  and students for an AI-augmented learning future.* JETELI 1(2).
  https://www.researchgate.net/publication/399573404 **[PDF in `docs/reference_papers/`]**
  (Confirm full author list against the PDF cover.)

## Metrics and evaluation

- **`cohen1968qwk`** Cohen (1968). *Weighted kappa.* Psychological Bulletin 70(4):213-220.
  https://doi.org/10.1037/h0026256
- **`williamson2012framework`** Williamson, Xi, Breyer (2012). *A framework for evaluation and use of
  automated scoring.* Educational Measurement: Issues and Practice 31(1):2-13.
  https://doi.org/10.1111/j.1745-3992.2011.00223.x

## Tools and models

- **`unsloth`** Unsloth AI (2024). *Unsloth: efficient LLM fine-tuning.*
  https://github.com/unslothai/unsloth
- **`gte2024`** Alibaba-NLP (2024). *GTE-multilingual-base (encoder pillar).*
  https://huggingface.co/Alibaba-NLP/gte-multilingual-base

---

### Removed in the audit
- `bergkirkpatrick2012` (statistical-significance testing) was removed: it was uncited after
  confidence intervals / significance tests were dropped from the thesis.

### Added in the audit (method papers the LLM pillar needed)
- `dettmers2023qlora` (QLoRA) and `hu2022lora` (LoRA): the LLM pillar fine-tunes with QLoRA, which
  was previously uncited.

### Note on metric comparability
QWK is dataset-dependent (class count, scale, rater noise). Cross-paper QWK/kappa values are
**contextual positioning, not a fair benchmark**. Alaoui et al. report **unweighted** Cohen's kappa
on a 3-class scale, which is not directly comparable to this thesis's 5-class QWK; the defensible
cross-study comparison is the **train/test generalisation gap** (Chapter 5).
