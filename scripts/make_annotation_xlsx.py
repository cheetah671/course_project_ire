"""Build Excel labelling sheets (one per annotator) from annotation/judge_validation_to_annotate.csv.

Each workbook has an "Instructions" sheet and a "Label" sheet with drop-down lists for the three labels, so values can
only be 0/1/2 (faithfulness, relevance) or 0/1 (correct). `judge_validation score` reads the .xlsx files directly.

Usage: python scripts/make_annotation_xlsx.py   -> annotation/ann_A.xlsx, ann_B.xlsx
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "annotation" / "judge_validation_to_annotate.csv"

INSTRUCTIONS = [
    ("What you are doing", "For each of the 150 rows, read the question, the system's answer and the passages it was given, "
                           "then fill the three yellow columns. Work alone: do not discuss items with the other annotator."),
    ("Order", "Read the question -> the system answer -> the passages. Use only the passages and the gold answers shown, "
              "not your own knowledge, for 'faithfulness'."),
    ("faithfulness (0/1/2)", "Is the system answer SUPPORTED BY THE PASSAGES?  2 = a passage states it (or it follows directly "
                             "from one or two passages).  1 = only partly supported, or needs an unstated assumption.  "
                             "0 = not supported, contradicted, a refusal, or 'I don't know'.  "
                             "Correctness does not matter here: a wrong answer that a passage states is still 2; "
                             "a correct answer that no passage supports is 0."),
    ("relevance (0/1/2)", "Does the answer ADDRESS THE QUESTION?  2 = the right kind of answer (a person for 'who', a date for "
                          "'when'; for zsRE 'subject | relation' rows, a value of the type the relation asks for).  "
                          "1 = related but off-target, hedged or too long.  0 = irrelevant, empty or a refusal."),
    ("correct (0/1)", "Is the answer CORRECT compared with the gold answers?  1 = matches a gold answer or is an unambiguous "
                      "equivalent ('JFK' = 'John F. Kennedy', 'President Obama' = 'Obama').  0 = otherwise.  If the gold "
                      "answer itself looks wrong, still label against it and write gold_issue in notes."),
    ("notes (optional)", "Free text. Useful tags: gold_issue, ambiguous_question, passage_truncated, multi_answer."),
    ("annotator", "Put your name in the 'annotator' column of the first row (or every row)."),
    ("Time", "About 1 minute per row, 2-3 hours in total. Save often. Do not change item_id or delete rows."),
    ("When both are done", "The third team member compares the two faithfulness columns and, where they differ, writes the "
                           "final value (0/1/2) in 'adjudicated_faithfulness' of ann_A.xlsx. Then run "
                           "judge_validation score (docs/05 §5) or ask Claude."),
]
COLUMNS = [  # (header, source column or None, width, editable)
    ("item_id", "item_id", 8, False), ("dataset", "dataset", 9, False),
    ("question_or_claim", "question_or_claim", 40, False), ("system_answer", "system_answer", 22, False),
    ("gold_answers", "gold_answers", 26, False),
    ("faithfulness_label", None, 12, True), ("relevance_label", None, 11, True), ("correct_label", None, 10, True),
    ("notes", None, 18, True), ("annotator_id", None, 11, True), ("adjudicated_faithfulness", None, 13, True),
    ("passages", "passages", 110, False),
]
YELLOW = PatternFill("solid", fgColor="FFF2CC")
HEAD = PatternFill("solid", fgColor="D9D9D9")


def build(df: pd.DataFrame, out: Path, who: str) -> None:
    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    ins["A1"], ins["A1"].font = f"Judge-validation labelling sheet: annotator {who}", Font(bold=True, size=13)
    for i, (k, v) in enumerate(INSTRUCTIONS, start=3):
        ins.cell(i, 1, k).font = Font(bold=True)
        ins.cell(i, 2, v).alignment = Alignment(wrap_text=True, vertical="top")
    ins.column_dimensions["A"].width, ins.column_dimensions["B"].width = 22, 120
    ws = wb.create_sheet("Label")
    for j, (h, _, w, edit) in enumerate(COLUMNS, start=1):
        c = ws.cell(1, j, h)
        c.font, c.fill = Font(bold=True), (YELLOW if edit else HEAD)
        c.alignment = Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[c.column_letter].width = w
    for i, row in enumerate(df.itertuples(index=False), start=2):
        r = row._asdict()
        for j, (h, src, _, edit) in enumerate(COLUMNS, start=1):
            c = ws.cell(i, j, (r[src] if src else None))
            c.alignment = Alignment(wrap_text=True, vertical="top")
            if edit:
                c.fill = YELLOW
        ws.row_dimensions[i].height = 150
    n = len(df) + 1
    for col, allowed in (("F", "0,1,2"), ("G", "0,1,2"), ("H", "0,1"), ("K", "0,1,2")):
        dv = DataValidation(type="list", formula1=f'"{allowed}"', allow_blank=True, showErrorMessage=True,
                            errorTitle="Invalid label", error=f"Choose one of {allowed}")
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n}")
    ws.freeze_panes = "C2"
    wb.active = 1
    wb.save(out)


def main():
    df = pd.read_csv(SRC, encoding="utf-8")
    for who in ("A", "B"):
        out = ROOT / "annotation" / f"ann_{who}.xlsx"
        build(df, out, who)
        print(f"wrote {out} ({len(df)} items)")


if __name__ == "__main__":
    main()
