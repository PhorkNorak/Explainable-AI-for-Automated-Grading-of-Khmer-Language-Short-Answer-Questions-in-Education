# Reference verification coverage

Audit dates: 2026-09-20 and 2026-09-21. Bibliography source files stay local with the manuscript; no active conference-paper source is confirmed. The historical audit below reviewed an earlier local bibliography. Historical citation keys are retained for identifying those checks. Their names are not evidence of authorship.

The [approved final report](<../reports/Explainable AI for Automated Grading of Khmer Language Short-Answer Questions in Education Report final.pdf>) is the current document reference. The checks below remain historical coverage records for the audit revision, not a fresh verification of the final PDF's entire bibliography. See [report versions and evidence](report_evidence.md). KhmerGrader is the model family covered by the final report.

## References checked against primary records

| Key | Correct work and primary source | Status and support boundary |
|---|---|---|
| `sung2022survey` | Haller, Aldea, Seifert, Strisciuglio (2022), [Survey on Automated Short Answer Grading with Deep Learning](https://arxiv.org/abs/2204.03503) | FIXED_AND_REVERIFIED: corrected misattributed authors. Record and abstract support the broad method taxonomy; no exhaustive review or individual numerical results checked. |
| `chang2024gpt4` | Kortemeyer (2023), [Performance of the Pre-Trained Large Language Model GPT-4 on Automated Short Answer Grading](https://arxiv.org/abs/2309.09338) | FIXED_AND_REVERIFIED: corrected author and year. Abstract supports the scoped SciEntsBank/Beetle comparison, not universal specialist superiority. |
| `kumar2020explainable` | Kumar and Boulanger (2020), [Explainable Automated Essay Scoring](https://www.frontiersin.org/journals/education/articles/10.3389/feduc.2020.572367/full) | VERIFIED: title, authors, year, journal, article identifier; inspected feature-based essay-scoring motivation and local/global explanation framing. Does not validate this project's Khmer reference-overlap metric or classroom effectiveness. The full article was not exhaustively audited. |
| `exasag2023` | Tornqvist, Mahamud, Mendez Guzman, Farazouli (2023), [ExASAG](https://aclanthology.org/2023.bea-1.29/) | VERIFIED for identity, authors, year, and abstract-level support: binary classifier, post-hoc attribution, human-annotation token overlap. Fine bibliographic pagination and full-text experiment details remain UNVERIFIED. |
| `devlin2019bert` | Devlin, Chang, Lee, Toutanova (2019), [BERT](https://aclanthology.org/N19-1423/) | VERIFIED: title/authors/venue/year/pages 4171-4186 and broad pretraining/fine-tuning description. Does not establish Khmer-specific performance. |
| `lundberg2017shap` | Lundberg and Lee (2017), [A Unified Approach to Interpreting Model Predictions](https://papers.nips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html) | VERIFIED for identity and SHAP motivation from official proceedings record; exact pagination and full-text derivation were not checked. |

These six keys were the active report citation set at the review checkpoint. Verification applies only to the stated fields and claims. It is not a claim that the entire bibliography or every sentence of a cited paper was verified.

## Model-card checks

The official [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Gemma4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), and [Gemma-SEA-LION-v4.5-E2B-IT](https://huggingface.co/aisingapore/Gemma-SEA-LION-v4.5-E2B-IT) records were opened on 2026-09-20. Their displayed licences were Apache-2.0, Apache-2.0, and MIT, respectively. Qwen is post-trained. SEA-LION identifies `google/gemma-4-E2B-it` as its upstream model. Gemma's card includes thinking-mode configuration, so the old assertion that only Qwen supports reasoning was removed. These checks do not identify the local adapter revision, verify publication of the report model weights, or certify downstream licence compliance.

## Unverified bibliography and unsuccessful checks

All other entries in the historical local bibliography are UNVERIFIED in this audit unless another audit record explicitly supplies a primary-source check and its coverage. They are an inherited reading list, not a verified literature review.

The [Pinto/Shin DOI](https://doi.org/10.1111/jedm.12438) and [Buoy DOI](https://doi.org/10.26599/TST.2021.9010060) did not resolve to readable content through the browsing tool. The [Arabic ASAG publisher page](https://ijece.iaescore.com/index.php/IJECE/article/view/31253) returned only a loading page. Their detailed claims remain UNVERIFIED; this audit did not inspect the Arabic PDF or verify its numerical comparison. The [Solano record](https://arxiv.org/abs/2507.05305) was reached but not read sufficiently to verify all metadata or temperature claims.

The previous assertion that every entry was web-verified is withdrawn. Comparing train/test gaps across unrelated studies does not cure differences in language, labels, dataset, or split design. No such gap is treated as a fair head-to-head comparison.
