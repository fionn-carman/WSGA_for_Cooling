#!/usr/bin/env python3
"""
Resolve compound CAS numbers and names from LB data to canonical SMILES.

Resolution order (CAS-first for reliability):
1. PubChem PUG REST by CAS number (~5 req/s)
2. PubChem PUG REST by name with variants (~5 req/s)
3. NCI CACTUS by CAS number (~1 req/s)
4. NCI CACTUS by name with variants (~1 req/s)
5. Unresolved -> lb_dc_unresolved.csv

Usage:
    python resolve_lb_smiles.py [--input lb_dc_25C.csv] [--output lb_dc_resolved.csv]
                                [--skip-pubchem] [--skip-cactus]
"""

import argparse
import asyncio
import logging
import re
from pathlib import Path
from urllib.parse import quote as url_quote

import httpx
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "lb_dc_25C.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "lb_dc_resolved.csv"
UNRESOLVED_OUTPUT = SCRIPT_DIR / "lb_dc_unresolved.csv"

PUBCHEM_PUG_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"
CACTUS_URL = "https://cactus.nci.nih.gov/chemical/structure/{name}/smiles"
PUBCHEM_CONCURRENCY = 5
PUBCHEM_DELAY = 0.2
CACTUS_CONCURRENCY = 2
CACTUS_DELAY = 1.0

MAX_MW = 500
CAS_PATTERN = re.compile(r"^\d{2,7}-\d{2}-\d$")


def validate_smiles(smiles: str) -> str | None:
    """Return canonical SMILES if valid, else None.

    Rejects salts and MW > 500.
    """
    if "." in smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if Descriptors.ExactMolWt(mol) > MAX_MW:
        return None
    return Chem.MolToSmiles(mol)


def clean_name(name: str) -> list[str]:
    """Generate cleaned name variants for lookup.

    Adapted from CRC pipeline's clean_crc_name().
    """
    variants = []
    base_name = name.strip()

    # Unicode Greek letters
    _UNICODE_GREEK = {
        "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
        "ε": "epsilon", "ω": "omega", "μ": "mu",
    }
    greek_ascii = base_name
    for uni, ascii_name in _UNICODE_GREEK.items():
        greek_ascii = greek_ascii.replace(uni + "-", ascii_name + "-")
        greek_ascii = greek_ascii.replace(uni, ascii_name + "-")
    if greek_ascii != base_name:
        variants.append(greek_ascii)
        base_name = greek_ascii

    # Strip stereo prefixes
    stereo_stripped = re.sub(r"^\([0-9RSEZ,]+\)-\s*", "", base_name, flags=re.IGNORECASE)
    if stereo_stripped != base_name:
        variants.append(stereo_stripped)

    ct_stripped = re.sub(r"^(?:cis|trans|meso|d,l|dl|rac)[,-]?\s*", "", base_name, flags=re.IGNORECASE)
    if ct_stripped != base_name:
        variants.append(ct_stripped)

    # Strip Greek letter prefixes
    greek_stripped = re.sub(
        r"^(?:alpha|beta|gamma|delta|epsilon|omega)-?\s*",
        "", base_name, flags=re.IGNORECASE,
    )
    if greek_stripped != base_name:
        variants.append(greek_stripped)

    # Handle parenthetical isomers
    paren_stripped = re.sub(r"\s*\([^)]*\)\s*$", "", base_name).strip()
    if paren_stripped != base_name and len(paren_stripped) > 3:
        variants.append(paren_stripped)

    # N-X or O-X prefix
    n_stripped = re.sub(r"^[NO]-", "", base_name)
    if n_stripped != base_name:
        variants.append(n_stripped)

    # Position numbering
    pos_stripped = re.sub(r"^[\d,]+-", "", base_name)
    if pos_stripped != base_name:
        variants.append(pos_stripped)

    # mono/di/tri prefixes
    for prefix in ("mono", "di", "tri", "tetra"):
        if base_name.lower().startswith(prefix):
            without = base_name[len(prefix):]
            if without and without[0] != "-":
                variants.append(without)
            elif without.startswith("-"):
                variants.append(without[1:])

    # Deduplicate
    seen = {name}
    deduped = []
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped


async def fetch_pubchem_smiles(
    client: httpx.AsyncClient,
    identifier: str,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Query PubChem PUG REST for a CAS number or compound name."""
    async with semaphore:
        url = PUBCHEM_PUG_URL.format(name=url_quote(identifier, safe=""))
        try:
            resp = await client.get(url, timeout=30)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
        except Exception:
            return None

        try:
            data = resp.json()
            props = data["PropertyTable"]["Properties"][0]
            smiles = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES")
            if not smiles:
                return None
        except (KeyError, IndexError, ValueError):
            return None

        await asyncio.sleep(PUBCHEM_DELAY)
        return validate_smiles(smiles)


async def fetch_cactus_smiles(
    client: httpx.AsyncClient,
    identifier: str,
    semaphore: asyncio.Semaphore,
) -> str | None:
    """Query NCI CACTUS for a CAS number or compound name."""
    async with semaphore:
        url = CACTUS_URL.format(name=url_quote(identifier, safe=""))
        try:
            resp = await client.get(url, timeout=30)
            if resp.status_code == 404:
                await asyncio.sleep(CACTUS_DELAY)
                return None
            resp.raise_for_status()
        except Exception:
            await asyncio.sleep(CACTUS_DELAY)
            return None

        text = resp.text.strip()
        await asyncio.sleep(CACTUS_DELAY)

        if not text:
            return None

        for line in text.splitlines():
            line = line.strip()
            if line:
                valid = validate_smiles(line)
                if valid:
                    return valid
        return None


async def resolve_compound(
    client_pubchem: httpx.AsyncClient,
    client_cactus: httpx.AsyncClient,
    sem_pubchem: asyncio.Semaphore,
    sem_cactus: asyncio.Semaphore,
    cas: str,
    name: str,
    skip_pubchem: bool = False,
    skip_cactus: bool = False,
) -> tuple[str | None, str]:
    """Resolve a single compound through all fallback methods.

    Returns (smiles, resolution_method).
    """
    has_cas = bool(cas) and CAS_PATTERN.match(cas.strip())
    names_to_try = [name] + clean_name(name)

    # 1. PubChem by CAS
    if not skip_pubchem and has_cas:
        smi = await fetch_pubchem_smiles(client_pubchem, cas, sem_pubchem)
        if smi:
            return smi, "pubchem_cas"

    # 2. PubChem by name variants
    if not skip_pubchem:
        for n in names_to_try:
            smi = await fetch_pubchem_smiles(client_pubchem, n, sem_pubchem)
            if smi:
                return smi, "pubchem_name"

    # 3. CACTUS by CAS
    if not skip_cactus and has_cas:
        smi = await fetch_cactus_smiles(client_cactus, cas, sem_cactus)
        if smi:
            return smi, "cactus_cas"

    # 4. CACTUS by name variants
    if not skip_cactus:
        for n in names_to_try:
            smi = await fetch_cactus_smiles(client_cactus, n, sem_cactus)
            if smi:
                return smi, "cactus_name"

    return None, "unresolved"


async def batch_resolve(
    compounds: list[dict],
    skip_pubchem: bool = False,
    skip_cactus: bool = False,
) -> list[dict]:
    """Resolve SMILES for all compounds."""
    sem_pubchem = asyncio.Semaphore(PUBCHEM_CONCURRENCY)
    sem_cactus = asyncio.Semaphore(CACTUS_CONCURRENCY)
    total = len(compounds)
    results = []
    method_counts = {}

    async with httpx.AsyncClient() as client_pubchem, httpx.AsyncClient() as client_cactus:
        for i, comp in enumerate(compounds):
            if (i + 1) % 50 == 0 or i == 0 or i == total - 1:
                log.info("  Resolution progress: %d / %d (%.0f%%)",
                         i + 1, total, (i + 1) / total * 100)

            smi, method = await resolve_compound(
                client_pubchem, client_cactus,
                sem_pubchem, sem_cactus,
                cas=str(comp.get("cas_number", "")),
                name=str(comp.get("name", "")),
                skip_pubchem=skip_pubchem,
                skip_cactus=skip_cactus,
            )
            results.append({**comp, "SMILES": smi, "resolution_method": method})
            method_counts[method] = method_counts.get(method, 0) + 1

    log.info("Resolution methods: %s", method_counts)
    return results


def main():
    parser = argparse.ArgumentParser(description="Resolve LB compounds to SMILES")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV")
    parser.add_argument("--skip-pubchem", action="store_true")
    parser.add_argument("--skip-cactus", action="store_true")
    args = parser.parse_args()

    if not Path(args.input).exists():
        log.error("Input not found: %s", args.input)
        return

    df = pd.read_csv(args.input)
    log.info("Loaded %d compounds from %s", len(df), args.input)

    # Only resolve compounds that have DC_25C
    has_dc = df["DC_25C"].notna()
    log.info("Compounds with DC at 25C: %d (will resolve these)", has_dc.sum())

    to_resolve = df[has_dc].to_dict("records")
    resolved = asyncio.run(batch_resolve(
        to_resolve,
        skip_pubchem=args.skip_pubchem,
        skip_cactus=args.skip_cactus,
    ))

    result_df = pd.DataFrame(resolved)

    # Summary
    has_smiles = result_df["SMILES"].notna()
    log.info("Total resolved: %d / %d (%.1f%%)",
             has_smiles.sum(), len(result_df),
             100 * has_smiles.sum() / len(result_df))

    # Save resolved
    result_df.to_csv(args.output, index=False)
    log.info("Saved resolved data to %s", args.output)

    # Save unresolved
    unresolved = result_df[~has_smiles]
    if len(unresolved) > 0:
        unresolved_cols = ["compound_no", "mol_formula", "name", "cas_number", "DC_25C"]
        unresolved[unresolved_cols].to_csv(UNRESOLVED_OUTPUT, index=False)
        log.info("Saved %d unresolved compounds to %s", len(unresolved), UNRESOLVED_OUTPUT)
        log.info("Unresolved compounds:")
        for _, row in unresolved.head(20).iterrows():
            log.info("  #%d %s [%s] DC=%.2f",
                     row["compound_no"], row["name"],
                     row.get("cas_number", ""), row["DC_25C"])
    else:
        log.info("All compounds resolved!")


if __name__ == "__main__":
    main()
