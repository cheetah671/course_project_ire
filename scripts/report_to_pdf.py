"""Render a Markdown report (with relative image paths) to PDF via a headless Chromium browser (Edge or Chrome).

Usage: python scripts/report_to_pdf.py docs/12_final_report.md [docs/12_final_report.pdf]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #111; }
h1 { font-size: 16pt; margin: 0 0 6pt; } h2 { font-size: 13pt; margin: 14pt 0 4pt; border-bottom: 1px solid #ccc; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.5pt; width: 100%; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; } img { max-width: 100%; page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 8.5pt; } p, li { margin: 3pt 0; }
"""

BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe", "msedge", "google-chrome", "chromium"]


def main():
    src = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src.with_suffix(".pdf")
    body = markdown.markdown(src.read_text(encoding="utf-8"), extensions=["tables", "fenced_code"])
    html = src.with_suffix(".print.html")  # next to the .md so relative image paths resolve
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>", encoding="utf-8")
    exe = next((b for b in BROWSERS if Path(b).exists() or shutil.which(b)), None)
    if exe is None:
        raise SystemExit("no Edge/Chrome found")
    # A separate profile stops an already-open browser window from taking over the job. The launcher can return
    # before the PDF is written, so delete the old PDF first and wait until the new one exists and stops growing;
    # otherwise a stale PDF (or a half-written one) passes silently.
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
        subprocess.run([exe, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                        f"--print-to-pdf={out}", html.as_uri()], check=True, timeout=120)
        last, deadline = -1, time.time() + 120
        while time.time() < deadline:
            size = out.stat().st_size if out.exists() else -1
            if size > 0 and size == last:
                break
            last = size
            time.sleep(1)
        else:
            raise SystemExit(f"browser did not finish writing {out}")
    html.unlink()
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
