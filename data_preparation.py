from pathlib import Path

import numpy as np
import pandas as pd


RENAME = {
    "number of citations": "citations",
    "number of studies examined": "studies",
    "Cochrane systematic review": "cochrane",
    "Other International review board or metastudy": "other_review",
}


def stevens_dictionary(raw):
    ratios = {"popularity", "number of studies examined", "number of citations",
              "no. of studies in Cochrane metastudy", "no. of studies in metastudy",
              "No. of individual studies / reviews", "N positive studies / trials",
              "% positive studies/ trials"}
    rows = []
    for col in raw.columns:
        scale, explanation = "nominale", "Tekst, emërtim, kategori ose adresë burimi."
        if raw[col].isna().all():
            scale, explanation = "e papërcaktuar", "Kolonë bosh; shkalla nuk mund të nxirret nga vlerat."
        elif col in ratios:
            scale, explanation = "raport", "Numërim ose përqindje me zero kuptimplote; kontrollo formatet tekstuale."
        elif col.startswith("evidence level"):
            scale, explanation = "ordinale", "Vlerësim 0–6; distancat ndërmjet niveleve nuk janë të garantuara të barabarta."
        elif col == "year(s) of metastudy":
            scale, explanation = "interval (pas parsimit)", "Vitet kalendarike janë interval; fusha mund të përmbajë disa vite ose '-'."
        elif col == "Cochrane systematic review":
            scale, explanation = "nominale", "Tregues binar i pranisë së rishikimit Cochrane: 0/1."
        elif col == "Other International review board or metastudy":
            scale, explanation = "raport", "Numër rishikimesh sipas burimit; mund të jetë mbi 1, nuk supozohet tregues binar."
        elif col == "OTW":
            explanation = "Etiketë burimore me kuptim të padokumentuar; mungesa nuk interpretohet si 0."
        rows.append({"column": col, "Stevens": scale, "explanation": explanation})
    return pd.DataFrame(rows)


def prepare_data(path):
    path = Path(path).expanduser().resolve()
    raw = pd.read_csv(path, encoding="utf-8")
    data = raw.copy()
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].str.strip().replace("", np.nan)
    dictionary = stevens_dictionary(data)
    quality = data.isna().sum().rename("missing").to_frame()
    evidence = [c for c in raw if c.startswith("evidence level")]
    if len(evidence) != 1:
        raise ValueError("Expected exactly one evidence-level column.")
    data = data.rename(columns={**RENAME, evidence[0]: "evidence"})
    numeric = ["popularity", "citations", "studies", "evidence", "cochrane", "other_review"]
    required = ["supplement", *numeric]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    for col in numeric:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    invalid = data[required].isna().any(axis=1)
    invalid |= ~np.isfinite(data[numeric]).all(axis=1)
    invalid |= (data[["popularity", "citations", "studies"]] < 0).any(axis=1)
    
    invalid |= ~data["evidence"].between(0, 6)
    invalid |= (data[["cochrane", "other_review"]] < 0).any(axis=1)
    for col in ["popularity", "citations", "studies", "cochrane", "other_review"]:
        invalid |= data[col].mod(1).ne(0)
    rejected = data.loc[invalid].copy()
    data = data.loc[~invalid].copy()
    duplicate_count = int(data.duplicated().sum())
    data = data.drop_duplicates()
    for col in ["popularity", "citations"]:
        conflicts = data.groupby("supplement")[col].nunique()
        if conflicts.gt(1).any():
            raise ValueError(f"Inconsistent {col} within supplements: {conflicts[conflicts.gt(1)].index.tolist()}")
    supplements = data.groupby("supplement", as_index=False).agg(
        popularity=("popularity", "first"), citations=("citations", "first"),
        mean_studies=("studies", "mean"),
    )
    if len(supplements) <= 3:
        raise ValueError("Nevojiten më shumë se 3 suplemente për modelin me dy shpjegues.")
    for col in ["popularity", "citations", "mean_studies"]:
        supplements[f"log_{col}"] = np.log1p(supplements[col])
    flags = supplements[["supplement"]].copy()
    for col in ["popularity", "citations", "mean_studies"]:
        q1, q3 = supplements[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        flags[f"{col}_IQR_flag"] = ~supplements[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    audit = {"raw_rows": len(raw), "raw_columns": len(raw.columns),
             "invalid_rows_removed": len(rejected), "exact_duplicates_removed": duplicate_count,
             "valid_claim_rows": len(data), "supplements": len(supplements),
             "empty_columns": raw.columns[raw.isna().all()].tolist()}
    return data, supplements, dictionary, quality, flags, rejected, audit
