#!/usr/bin/env python3
"""Extract text from PDFs (recursively) and save to .txt files.

Tries to use system `pdftotext` (poppler) if available, otherwise falls back to PyPDF2.

Usage:
  python3 pdf_to_txt.py /path/to/search --out /path/to/output

If --out is omitted, .txt files are written next to each PDF.
"""
import shutil
import subprocess
from pathlib import Path

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None


def pdftotext_available():
    return shutil.which("pdftotext") is not None


def extract_with_pdftotext(pdf_path: Path, txt_path: Path) -> bool:
    try:
        # Use -layout to preserve columns/rough layout. Remove if you don't want it.
        subprocess.run(["pdftotext", "-layout", str(pdf_path), str(txt_path)], check=True)
        return True
    except Exception:
        return False


def extract_with_pypdf2(pdf_path: Path, txt_path: Path) -> bool:
    if PdfReader is None:
        return False
    try:
        reader = PdfReader(str(pdf_path))
        texts = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                texts.append(t)
        txt_path.write_text("\n\n".join(texts), encoding="utf-8")
        return True
    except Exception:
        return False


def convert_all_pdfs(root: Path, out_dir: Path = None):
    root = root.resolve()
    if out_dir:
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
    use_pdftotext = pdftotext_available()
    if use_pdftotext:
        print("Using system pdftotext for extraction.")
    elif PdfReader is not None:
        print("pdftotext not found; using PyPDF2.")
    else:
        print("No extraction backend available. Install poppler (pdftotext) or PyPDF2.")
        return

    pdf_files = list(root.rglob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in", root)
        return

    for pdf in pdf_files:
        rel = pdf.relative_to(root)
        if out_dir:
            txt_path = out_dir.joinpath(rel.with_suffix(".txt"))
            txt_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            txt_path = pdf.with_suffix(".txt")

        print(f"Processing: {pdf} -> {txt_path}")
        ok = False
        if use_pdftotext:
            ok = extract_with_pdftotext(pdf, txt_path)
        if not ok:
            ok = extract_with_pypdf2(pdf, txt_path)
        if not ok:
            print(f"Failed to extract: {pdf}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract text from PDFs to .txt files.")
    p.add_argument("path", nargs="?", default=".", help="Directory to scan for PDFs (default: current dir).")
    p.add_argument("--out", "-o", help="Optional output directory for .txt files.")
    args = p.parse_args()

    root = Path(args.path)
    out = Path(args.out) if args.out else None
    convert_all_pdfs(root, out)
