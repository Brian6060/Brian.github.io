#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
if command -v xelatex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode -halt-on-error B0_Report__seed42.tex
  xelatex -interaction=nonstopmode -halt-on-error B0_Report__seed42.tex
elif command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error B0_Report__seed42.tex
  pdflatex -interaction=nonstopmode -halt-on-error B0_Report__seed42.tex
else
  echo "No TeX engine found (xelatex/pdflatex)."
  exit 2
fi
