#!/usr/bin/env python3
"""
Extract dielectric constant data from the Landolt-Börnstein Pure Liquids PDF.

Parses the organic liquids section (2.1.3, pages 15-228) using pdfplumber
text extraction with line-based pattern matching:
  - Compound headers: No. + formula + name + [CAS]
  - Single-temperature values: e (T K) value
  - T/ε pair rows: T [K] row followed by e (T) row

Output: lb_dc_raw.csv with columns:
    compound_no, mol_formula, name, cas_number, T_K, epsilon, ref_code

Usage:
    python extract_lb_dc.py [--pdf PATH] [--output PATH] [--start-page 14] [--end-page 228]
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import pdfplumber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PDF = Path.home() / "Downloads" / "LB_2 Pure Liquids_ Data.pdf"
DEFAULT_OUTPUT = SCRIPT_DIR / "lb_dc_raw.csv"

# Regex patterns
CAS_PATTERN = re.compile(r"\[(\d{2,7}-\d{2}-\d)\]")
# Compound header: starts with integer, has text, has [CAS]
HEADER_WITH_CAS = re.compile(r"^(\d{1,4})\s+(.+?)\s*\[(\d{2,7}-\d{2}-\d)\]\s*$")
# Header start without CAS on same line (multi-line header)
HEADER_START = re.compile(r"^(\d{1,4})\s+[A-Z]")
# CAS on a continuation line (no leading compound number)
CAS_CONTINUATION = re.compile(r"^[A-Za-z0-9,.\s()-]+\[(\d{2,7}-\d{2}-\d)\]")
# Single epsilon: e (T K) value
SINGLE_EPS = re.compile(r"^e\s*\(\s*([\d.]+)\s*K\s*\)\s+([\d.]+)")
# T [K] row
T_ROW = re.compile(r"^T\s*\[K\]\s+([\d.\s]+)")
# e(T) row
EPS_ROW = re.compile(r"^e\s*\(T\)\s+([\d.\s]+)")
# Ref code at end of line: 1-2 digit number + letter + number
REF_PATTERN = re.compile(r"\s+(\d{1,2}\s+[A-Z]\s+\d)\s*$")
# Page header patterns
PAGE_HEADER = re.compile(
    r"^(Ref\.\s*p\.\s*229|.*Pure liquids.*data|No\.\s+Gross|formula|_{10,})"
)
# Subscript-only line (digits and spaces)
SUBSCRIPT_LINE = re.compile(r"^[\d\s]+$")
# Continuation value line: just a number + optional ref code
CONTINUATION_VALUE = re.compile(r"^\s*([\d.]+)\s+(.*)$")


def parse_float(s: str) -> float | None:
    """Parse a numeric string."""
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_ref_and_values(text: str) -> tuple[list[float], str]:
    """Split a data string into numeric values and a ref code.

    Ref codes are at the end and contain letters (e.g., '74 W 2').
    Returns (values, ref_code).
    """
    tokens = text.strip().split()
    if not tokens:
        return [], ""

    # Walk backwards to find where ref code starts
    # Ref codes contain letters; data values are pure numbers
    ref_start = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        if parse_float(tokens[i]) is None:
            # This token contains letters -> part of ref
            ref_start = i
        else:
            # Pure number. Check if it's part of ref or data.
            # Ref numbers are small (1-99) and preceded by more ref tokens
            if i > 0 and ref_start < len(tokens) and ref_start == i + 1:
                # This number immediately precedes a letter token -> ref number
                val = parse_float(tokens[i])
                if val is not None and val < 100 and val == int(val):
                    ref_start = i
                    continue
            break

    data_tokens = tokens[:ref_start]
    ref_tokens = tokens[ref_start:]

    values = []
    for t in data_tokens:
        v = parse_float(t)
        if v is not None:
            values.append(v)

    ref_code = " ".join(ref_tokens)
    return values, ref_code


def extract_all_lines(pdf_path: str, start_page: int, end_page: int) -> list[str]:
    """Extract all text lines from the PDF page range."""
    pdf = pdfplumber.open(pdf_path)
    total_pages = min(len(pdf.pages), end_page)
    all_lines = []

    for pg_idx in range(start_page, total_pages):
        page = pdf.pages[pg_idx]
        text = page.extract_text()
        if text:
            for line in text.split("\n"):
                all_lines.append(line)

    pdf.close()
    log.info("Extracted %d text lines from pages %d-%d", len(all_lines), start_page + 1, total_pages)
    return all_lines


def parse_lines(lines: list[str]) -> list[dict]:
    """Parse extracted text lines into T/ε records using a state machine."""
    records = []

    # State
    current_compound = None  # {compound_no, mol_formula, name, cas_number}
    pending_header_no = None  # compound number waiting for CAS on next line
    pending_header_text = ""  # accumulated header text
    pending_temps = None  # (temps_list, ref_code) waiting for ε(T) row
    last_single_eps_temp = None  # temperature from last ε(T K) line (for continuation)

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Skip page headers and separators
        if PAGE_HEADER.match(line_stripped):
            continue
        if line_stripped.startswith("_"):
            continue

        # --- Try: compound header (number + content + [CAS] on one line) ---
        m_header = HEADER_WITH_CAS.match(line_stripped)
        if m_header:
            compound_no = int(m_header.group(1))
            middle = m_header.group(2).strip()
            cas = m_header.group(3)

            # Extract name: strip formula chars from the start of middle
            # Formula chars: uppercase letter followed by lowercase/digits, spaces between elements
            # Name typically starts with a lowercase letter or digit prefix like "1,2-"
            name = _extract_name(middle)

            current_compound = {
                "compound_no": compound_no,
                "mol_formula": "",  # we don't need exact formula
                "name": name,
                "cas_number": cas,
            }
            pending_header_no = None
            pending_header_text = ""
            pending_temps = None
            last_single_eps_temp = None
            continue

        # --- Try: header start (number + formula, no CAS yet) ---
        m_start = HEADER_START.match(line_stripped)
        if m_start and not line_stripped.startswith("T "):
            # Check it's not a data line mistakenly matching
            # If line contains [CAS], handle above already caught it
            cas_m = CAS_PATTERN.search(line_stripped)
            if cas_m:
                # CAS is present but HEADER_WITH_CAS didn't match — extract manually
                compound_no = int(m_start.group(1))
                before_cas = line_stripped[:cas_m.start()].strip()
                # Remove compound number prefix
                after_no = re.sub(r"^\d{1,4}\s+", "", before_cas)
                name = _extract_name(after_no)
                current_compound = {
                    "compound_no": compound_no,
                    "mol_formula": "",
                    "name": name,
                    "cas_number": cas_m.group(1),
                }
                pending_header_no = None
                pending_header_text = ""
                pending_temps = None
                last_single_eps_temp = None
                continue
            else:
                # Multi-line header: save number, wait for CAS
                pending_header_no = int(m_start.group(1))
                pending_header_text = line_stripped
                continue

        # --- Try: CAS continuation (CAS on a line without compound number) ---
        if pending_header_no is not None:
            cas_m = CAS_PATTERN.search(line_stripped)
            if cas_m:
                # Found the CAS line for the pending header
                before_cas = line_stripped[:cas_m.start()].strip()
                # Combine with accumulated header text for name extraction
                combined = pending_header_text + " " + before_cas
                # Remove compound number from start
                combined = re.sub(r"^\d{1,4}\s+", "", combined)
                name = _extract_name(combined)

                current_compound = {
                    "compound_no": pending_header_no,
                    "mol_formula": "",
                    "name": name,
                    "cas_number": cas_m.group(1),
                }
                pending_header_no = None
                pending_header_text = ""
                pending_temps = None
                last_single_eps_temp = None
                continue
            elif SUBSCRIPT_LINE.match(line_stripped):
                # Subscript line for formula — skip but keep pending
                continue
            elif not line_stripped[0].isdigit() and not line_stripped.startswith("e") and not line_stripped.startswith("T"):
                # More formula/name text
                pending_header_text += " " + line_stripped
                continue
            else:
                # Something else — give up on this header
                pending_header_no = None
                pending_header_text = ""

        # Skip if no current compound
        if current_compound is None:
            continue

        # --- Try: single epsilon e(T K) value ---
        m_eps = SINGLE_EPS.match(line_stripped)
        if m_eps:
            pending_temps = None
            temp_k = parse_float(m_eps.group(1))
            eps_val = parse_float(m_eps.group(2))
            if temp_k is not None and eps_val is not None:
                # Get ref code from remainder
                remainder = line_stripped[m_eps.end():]
                _, ref_code = extract_ref_and_values(remainder)
                records.append({
                    **current_compound,
                    "T_K": temp_k,
                    "epsilon": eps_val,
                    "ref_code": ref_code,
                })
                last_single_eps_temp = temp_k
            continue

        # --- Try: T [K] row ---
        m_t = T_ROW.match(line_stripped)
        if m_t:
            last_single_eps_temp = None
            data_part = line_stripped[len("T [K]"):].strip()
            values, ref_code = extract_ref_and_values(data_part)
            if values:
                pending_temps = (values, ref_code)
            continue

        # --- Try: e(T) row ---
        m_e = EPS_ROW.match(line_stripped)
        if m_e:
            last_single_eps_temp = None
            data_part = line_stripped[m_e.start() + len(m_e.group(0)) - len(m_e.group(0)):].strip()
            # Re-extract properly
            after_prefix = re.sub(r"^e\s*\(T\)\s*", "", line_stripped)
            values, _ = extract_ref_and_values(after_prefix)
            if values and pending_temps is not None:
                temps, ref_code = pending_temps
                n_pairs = min(len(temps), len(values))
                for i in range(n_pairs):
                    records.append({
                        **current_compound,
                        "T_K": temps[i],
                        "epsilon": values[i],
                        "ref_code": ref_code,
                    })
                pending_temps = None
            continue

        # --- Try: continuation value (for single epsilon at same T) ---
        if last_single_eps_temp is not None:
            m_cont = CONTINUATION_VALUE.match(line_stripped)
            if m_cont:
                val = parse_float(m_cont.group(1))
                if val is not None:
                    remainder = m_cont.group(2) if m_cont.group(2) else ""
                    _, ref_code = extract_ref_and_values(remainder)
                    records.append({
                        **current_compound,
                        "T_K": last_single_eps_temp,
                        "epsilon": val,
                        "ref_code": ref_code,
                    })
                    continue

        # --- Subscript line or unrecognised — skip ---
        if SUBSCRIPT_LINE.match(line_stripped):
            continue

    return records


def _extract_name(text: str) -> str:
    """Extract compound name from header text (after compound number, before CAS).

    The text contains formula fragments and the name. The name typically starts
    with a lowercase letter, digit, or special char like '(' after the formula.
    Formula chars are uppercase letters followed by optional lowercase + digits.
    """
    text = text.strip()

    # Remove subscript digits that got merged
    # Formula pattern: sequences of [A-Z][a-z]?\d* separated by spaces
    # Name starts with: lowercase, digit, '(', or common prefixes

    # Strategy: find the first word that looks like a name (not a formula element)
    # Formula elements: single uppercase or Uppercase+lowercase (e.g., C, H, O, N, Cl, Br, Si)
    # Name words: start with lowercase, or are common prefixes (1,2- etc.), or known name starts

    tokens = text.split()
    name_start = 0

    for i, token in enumerate(tokens):
        # Formula element: all-uppercase single letter or Xx pattern, possibly with digits
        # But be careful: "N," is a name part (as in "N,N-dimethyl...")
        clean = re.sub(r"\d+", "", token)
        if re.match(r"^[A-Z][a-z]?$", clean) and "," not in token and "-" not in token:
            name_start = i + 1
            continue
        # Still formula: just digits (subscripts that got merged)
        if re.match(r"^\d+$", token):
            name_start = i + 1
            continue
        # This looks like a name token
        break

    name = " ".join(tokens[name_start:]).strip()
    # Clean up
    name = re.sub(r"\s+", " ", name)
    return name


def clean_records(records: list[dict]) -> pd.DataFrame:
    """Clean and validate extracted records."""
    if not records:
        log.warning("No records to clean")
        return pd.DataFrame(columns=[
            "compound_no", "mol_formula", "name", "cas_number",
            "T_K", "epsilon", "ref_code",
        ])

    df = pd.DataFrame(records)
    n_raw = len(df)

    # Validate temperature range
    valid_t = (df["T_K"] > 10) & (df["T_K"] < 700)
    df = df[valid_t].copy()

    # Validate epsilon range
    valid_eps = (df["epsilon"] > 0) & (df["epsilon"] < 200000)
    df = df[valid_eps].copy()

    log.info("Cleaned: %d -> %d records (removed %d)",
             n_raw, len(df), n_raw - len(df))
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Extract LB dielectric constant data from PDF"
    )
    parser.add_argument("--pdf", default=str(DEFAULT_PDF), help="Path to LB PDF")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    parser.add_argument("--start-page", type=int, default=14,
                        help="Start page (0-indexed, default 14 = page 15)")
    parser.add_argument("--end-page", type=int, default=228,
                        help="End page (exclusive, default 228)")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        log.error("PDF not found: %s", args.pdf)
        return

    lines = extract_all_lines(args.pdf, args.start_page, args.end_page)
    records = parse_lines(lines)
    log.info("Total raw records extracted: %d", len(records))

    df = clean_records(records)

    if len(df) == 0:
        log.error("No valid records extracted!")
        return

    # Summary statistics
    log.info("Temperature range: %.1f - %.1f K", df["T_K"].min(), df["T_K"].max())
    log.info("Epsilon range: %.4f - %.4f", df["epsilon"].min(), df["epsilon"].max())
    log.info("Unique compounds: %d", df["compound_no"].nunique())
    log.info("Compounds with CAS: %d",
             df[df["cas_number"] != ""]["compound_no"].nunique())

    # Spot check
    log.info("First 10 records:")
    for _, row in df.head(10).iterrows():
        log.info("  #%d %s | T=%.1f | eps=%.4f | CAS=%s",
                 row["compound_no"], row["name"],
                 row["T_K"], row["epsilon"], row["cas_number"])

    df.to_csv(args.output, index=False)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
