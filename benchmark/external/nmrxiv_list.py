"""List nmrXiv's openly licensed 1D ¹H datasets, as candidates for the benchmark.

Reads the public listing API (projects, samples, datasets), joins each dataset to its
project's licence and its sample's metadata, and writes one row per 1D ¹H dataset whose
project allows redistribution with attribution (CC BY, CC0, CDLA-Permissive).

    python benchmark/external/nmrxiv_list.py CACHE_DIR [OUT_CSV]

CACHE_DIR holds the downloaded listing pages (fetched if missing). Nothing here downloads
spectra; the list says where each one is.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import urllib.request
from pathlib import Path

API = "https://nmrxiv.org/api/v1/list"
OPEN_LICENCES = ("CC BY 4.0", "CC BY 3.0", "CC0 1.0", "CDLA-Permissive")


def pages(cache: Path, kind: str) -> list[dict]:
    out, page, last = [], 1, 1
    while page <= last:
        path = cache / f"{kind}-{page}.json"
        if not path.is_file():
            with urllib.request.urlopen(f"{API}/{kind}?page={page}", timeout=60) as r:
                path.write_bytes(r.read())
        doc = json.loads(path.read_text(encoding="utf-8"))
        last = doc.get("meta", {}).get("last_page") or doc.get("last_page") or 1
        out += doc["data"]
        page += 1
    return out


def short_licence(lic: object) -> str:
    title = lic.get("title", "") if isinstance(lic, dict) else str(lic or "")
    m = re.search(r"\(([^)]*)\)\s*$", title)
    return m.group(1) if m else title


def field(desc: str, name: str) -> str:
    m = re.search(rf"\b{name}: ([^\s<]+)", desc or "")  # values can end in "</br>"
    return m.group(1) if m else ""


def ident(url: str | None) -> str:
    return (url or "").rstrip("/").rsplit("/", 1)[-1]


def is_1d_proton(ds: dict, sample: dict) -> bool:
    kind = (ds.get("type") or "").lower()
    if re.search(r"cosy|hsqc|hmbc|noesy|roesy|tocsy|2d|13c|dept|19f|31p|29si|15n", kind):
        return False
    if "1h" in kind:
        return True
    desc = sample.get("description", "")
    return field(desc, "nucleus") == "1H" and field(desc, "dimension") == "1"


def main(cache: Path, out: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    projects = {ident(p["public_url"]): p for p in pages(cache, "projects")}
    samples = {ident(s["public_url"]): s for s in pages(cache, "samples")}
    rows = []
    for ds in pages(cache, "datasets"):
        project = projects.get(ident((ds.get("project") or {}).get("public_url")))
        sample = samples.get(ident((ds.get("study") or {}).get("public_url")), {})
        if not project or not project.get("is_public"):
            continue
        licence = short_licence(project.get("license"))
        if not licence.startswith(OPEN_LICENCES) or not is_1d_proton(ds, sample):
            continue
        desc = sample.get("description", "")
        mol = (sample.get("molecules") or [{}])[0]
        name = ds.get("name") or ""
        rows.append({
            "dataset": ds.get("identifier"),
            "dataset_url": ds.get("public_url"),
            "name": name,
            "format": "jcamp" if name.lower().endswith((".jdx", ".dx")) else "raw",
            "sample": sample.get("identifier", ""),
            "molecule_formula": mol.get("molecular_formula") or "",
            "molecule_smiles": mol.get("canonical_smiles") or "",
            "n_molecules": len(sample.get("molecules") or []),
            "solvent": field(desc, "solvent"),
            "field_mhz": field(desc, "baseFrequency").split(",")[0],  # ¹H first in 2D
            "pulse": field(desc, "pulseSequence"),
            "project": project.get("identifier"),
            "project_name": project.get("name"),
            "project_doi": project.get("doi") or "",
            "licence": licence,
        })  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["dataset"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} datasets from {len({r['project'] for r in rows})} projects -> {out}")


if __name__ == "__main__":
    here = Path(__file__).parent
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else here / "nmrxiv_1h.csv")
