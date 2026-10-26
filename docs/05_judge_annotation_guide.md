# Deliverable 4: 150-item judge-validation annotation guide

**Purpose.** Before trusting the LLM judge (RAGAS faithfulness, answer relevancy), measure whether it
agrees with humans on our data. The same labels also measure how strict EM is.

## 1. Items
- 150 items from the **dev** split (test stays untouched by human inspection), produced by
  `python -m ragroute.eval.judge_validation sample`.
- Stratified by dataset × (EM correct / incorrect) × (k = 3 / 10), one config per query within a
  stratum, so easy and hard, short and long contexts all appear.
- Output: `annotation/judge_validation_to_annotate.csv` (template with the same columns:
  `annotation/judge_validation_template.csv`).

## 2. Who and how
- Two annotators label all 150 items independently (≈ 2–3 h each). Do not discuss items before both finish.
- A third team member adjudicates disagreements on faithfulness in `adjudicated_faithfulness`.
- Annotators **must not see** the judge scores, the config id, or EM/F1. The sheet hides them.
- Read the question, then the system answer, then the passages. Use only the passages and gold
  answers shown, not outside knowledge, for faithfulness.

## 3. Labels

### faithfulness_label: is the system answer supported by the passages shown?
| Label | Meaning | Example |
|---|---|---|
| 2 | Fully supported: a passage states it, or it follows directly from one or two passages | Q: "Who wrote X?" Passage: "X is a novel by Jane Doe." A: "Jane Doe" |
| 1 | Partly supported or ambiguous: part of the answer is supported, or support needs an unstated assumption | A: "Jane Doe, 1923" where only the author is in the passages |
| 0 | Not supported or contradicted: nothing in the passages supports it, it is a refusal or "I don't know", or the passages say otherwise | A: "John Smith" |
Rules: correctness does not matter here. A wrong answer that a passage states is still 2 (faithful to
a misleading passage). A correct answer that no passage supports is 0 (from parametric memory).

### relevance_label: does the answer address the question asked?
2 = directly answers the question type asked (a person for "who", a date for "when"); 1 = related but
off-target, hedged, or over-long; 0 = irrelevant, empty, or a refusal.
For FEVER: 2 = a valid verdict (SUPPORTS/REFUTES); 0 = anything else.

### correct_label: is the answer correct with respect to the gold answers?
1 = matches a gold answer or is an unambiguous equivalent (alias, different but correct formatting
"JFK" vs "John F. Kennedy", added title "President Obama"); 0 = otherwise. When gold answers look
wrong or outdated, label against gold anyway and write `gold_issue` in `notes`.

### notes
Free text. Use these tags where they apply: `gold_issue`, `ambiguous_question`, `passage_truncated`, `multi_answer`.

## 4. Analysis (`python -m ragroute.eval.judge_validation score --ann A.csv B.csv --judge ...`)
- Inter-annotator agreement: Cohen's κ (and quadratic-weighted κ for the 3-point scales) with 95% bootstrap CIs.
- Judge vs adjudicated human: Spearman and Pearson (faithfulness score vs 0/1/2), Cohen's κ with the
  judge binarised at 0.5 and humans binarised as label 2 vs {0, 1}; relevancy Spearman.
- EM vs human correctness: κ and **EM false-negative rate** (human-correct but EM = 0); F1 vs human (Spearman).
- **Decision rule (pre-registered):** RAGAS is used at scale only if judge–human faithfulness κ ≥ 0.6
  (team to confirm) and the Spearman CI lower bound > 0.4. Otherwise RAGAS appears only as a
  diagnostic, with the agreement numbers next to it.
- Report both judge–human and human–human agreement: if humans agree at κ = 0.7 and the judge at
  0.65, the judge is near the ceiling of what the labels support.

## 5. Spreadsheet template
`annotation/judge_validation_template.csv` columns:
`item_id, dataset, question_or_claim, gold_answers, passages, system_answer, faithfulness_label,
relevance_label, correct_label, notes, annotator_id, adjudicated_faithfulness`.
Open in Google Sheets or Excel; add data validation (lists 0/1/2, 0/1/2, 0/1) to the three label columns.
