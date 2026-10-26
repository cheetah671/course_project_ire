# Which RAG Configuration for Which Query? Cost-Aware Per-Query Configuration Routing

IRE course project. We run **every** RAG configuration (24) on **every** query, which turns the grid
into oracle labels: for each query we know which configs succeed and what each costs. We then ask how
much of the oracle's cost-quality headroom a router recovers over the best static config, from the
question alone (Router A) and with cheap post-retrieval signals (Router B).

| Document | Contents |
|---|---|
| [docs/00_assumptions_open_items.md](docs/00_assumptions_open_items.md) | Assumptions, open items, the unresolved "Bllm" term |
| [docs/01_proposal.md](docs/01_proposal.md) | Deliverable 1: proposal / report skeleton |
| [docs/02_experiment_design.md](docs/02_experiment_design.md) | Full design with Decision → Alternatives → Why → Risk records; budget arithmetic |
| [docs/03_repo_and_harness.md](docs/03_repo_and_harness.md) | Deliverable 2: repo layout, commands, AutoRAG notes and workarounds |
| [docs/04_slurm_ada.md](docs/04_slurm_ada.md) | Deliverable 3: running on Ada |
| [docs/05_judge_annotation_guide.md](docs/05_judge_annotation_guide.md) | Deliverable 4: 150-item judge validation guide ([template](annotation/judge_validation_template.csv)) |
| [docs/06_tables_figures_report_template.md](docs/06_tables_figures_report_template.md) | Deliverable 5: exact tables/figures, analysis plan, final report template |
| [docs/07_plan_and_team_split.md](docs/07_plan_and_team_split.md) | Deliverable 6: week-by-week plan, 3-person split |
| [docs/08_risks.md](docs/08_risks.md) | Deliverable 7: risk register |
| [docs/09_requirements_checklist.md](docs/09_requirements_checklist.md) | Deliverable 8: requirement → where satisfied; open items |

## Quick start

```bash
# local (analysis only): tests + synthetic dry run of router/tables/figures (synthetic output is FAKE)
pip install -r requirements-analysis.txt
pytest
python -m ragroute.synthetic --out synthetic_run

# Ada (see docs/04_slurm_ada.md)
bash slurm/00_setup_env.sh          # once, login node
bash scripts/run_pipeline.sh        # corpus -> retrieval/rerank -> generation -> collect/router/tables/figures
```

Pipeline order: `prepare_qa → mine_candidates → build_corpus → splits → make_autorag_yaml → run_stages
retrieval → run_stages rerank → build_features → run_stages generate (×2 generators) → collect →
train_eval → tables → plots`, plus side studies: `check_equivalence`, `latency`, `judge_validation`, `ragas_eval`.

No numbers in these documents are results. Every result placeholder is written as `[[...]]`.
