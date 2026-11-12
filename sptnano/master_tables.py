"""Step 2c: merge several experiment folders into one instant table and one window table.

Per folder (kept separate until the end):
  load CSVs -> fix known wrong condition/filename -> parse metadata from filename + folder -> check
Then: merge (diagonal_relaxed) -> HTTCAG72 -> HTT72CAG -> rebuild short unique_id and matching window_uid
-> verify -> save parquet.

Dataset-specific choices (which folders, replicate per folder, corrections) live in configs/master_tables.yaml
and are passed around as `rules`.

unique_id = uid_<loc><type0><mol0><geno0><group0>_p<particle>_r<replicate>_<hash of folder|condition|filename|particle|replicate>
window_uid = unique_id + "_" + <time_window>_<frame_start>_<frame_end> taken from the original window_uid.
"""

from __future__ import annotations

import hashlib
import os
import re
import time

import polars as pl

MISSING_VALUE = "Unknown"
KEY_COLS = ["condition", "filename"]

FILENAME_KEY_TO_COLUMN = {
    "loc": "location",
    "type": "type",
    "mol": "mol",
    "geno": "geno",
    "cas": "cas",
    "dox": "dox",
    "sur": "sur",
    "dday": "dday",
    "rep": "replicate",
    "cell": "cell",
}
SPECIAL_FILENAME_KEYS = {"noco"}
TYPE_TO_CELL = {"vent": "neuron", "cort": "neuron", "ES": "ES"}
PREP_TO_TYPE = {"ventral": "vent", "cortical": "cort"}
WILDTYPE_CAG = "20"

IDENTIFIER_COLUMNS = [
    "group", "experiment_date", "location", "cell", "type", "mol", "geno", "cas", "dox",
    "hap40", "sur", "dday", "nocodazole_concentration", "replicate", "movie_number",
]
METADATA_COLUMNS = IDENTIFIER_COLUMNS + ["identifier"]
CHECK_COLUMNS = ["check_conflicts", "check_unrecognized_keys", "check_duplicate_keys", "check_overrides"]
LEGACY_COLUMN_SOURCES = {"molecule": "mol", "genotype": "geno", "cell_type": "cell"}

KV_TOKEN_RE = re.compile(r"^([A-Za-z]+)-(.+)$")
MOVIE_NUMBER_RE = re.compile(r"_(\d+)(?:_cropped)?(?:_tracked)?$")
ISSUE_SCHEMA = {"folder": pl.Utf8, "issue": pl.Utf8, "filename": pl.Utf8, "detail": pl.Utf8}


# ----------------------------------------------------------------------------- loading

def load_folder_tables(master_dir, folders, read_csv_kwargs=None):
    """Read saved_data/instant_df_processed.csv and time_windowed_df_processed.csv for each folder."""
    read_csv_kwargs = read_csv_kwargs or {"infer_schema_length": 10000}
    instant_by_folder, windowed_by_folder = {}, {}
    for folder in folders:
        saved_data = os.path.join(master_dir, folder, "saved_data")
        for name, store in (("instant_df_processed", instant_by_folder), ("time_windowed_df_processed", windowed_by_folder)):
            path = os.path.join(saved_data, f"{name}.csv")
            if not os.path.exists(path):
                print(f"  MISSING {path}")
                continue
            start = time.time()
            store[folder] = pl.read_csv(path, **read_csv_kwargs)
            print(f"  {folder}/{name}.csv: {store[folder].shape} ({time.time() - start:.0f}s)")
    return instant_by_folder, windowed_by_folder


# ----------------------------------------------------------------------------- metadata parsing

def extract_metadata_from_foldername(foldername):
    """group-HTT77_something -> HTT77."""
    match = re.search(r"group-([^_]+)", foldername)
    return match.group(1) if match else "Unknown"


def create_identifier_polars(cell, type_val, mol, geno, group):
    """First letter of cell, type, mol, geno, group (x if unknown)."""
    parts = [cell, type_val, mol, geno, group]
    return "".join([str(part)[0].lower() if part and str(part) != "Unknown" else "x" for part in parts])


def extract_nocodazole_concentration(filename):
    match = re.search(r"noco-([0-9]+ngml)", str(filename), flags=re.IGNORECASE)
    return match.group(1) if match else "none"


def parse_folder_metadata(folder_basename):
    """group, experiment_date, prep (ventral/cortical) and Q-length from the folder name."""
    group = extract_metadata_from_foldername(folder_basename)
    date_match = re.match(r"^(\d{1,2})[-_](\d{1,2})[-_](\d{4})", folder_basename)
    if date_match:
        month, day, year = (int(part) for part in date_match.groups())
        experiment_date = f"{year:04d}-{month:02d}-{day:02d}"
    else:
        experiment_date = MISSING_VALUE
    lowered = folder_basename.lower()
    folder_prep = "ventral" if "ventral" in lowered else "cortical" if "cortical" in lowered else MISSING_VALUE
    q_match = re.search(r"Q(\d+)", group)
    return {
        "group": group,
        "experiment_date": experiment_date,
        "folder_prep": folder_prep,
        "group_q": q_match.group(1) if q_match else None,
    }


def is_missing(value):
    return value is None or str(value).strip() == "" or value == MISSING_VALUE


def parse_key_value_tokens(text):
    """Return ({key: [values...]}, [plain tokens]) from an underscore-separated string."""
    values, plain_tokens = {}, []
    for token in str(text).split("_"):
        match = KV_TOKEN_RE.match(token)
        if match:
            values.setdefault(match.group(1).lower(), []).append(match.group(2))
        elif token:
            plain_tokens.append(token)
    return values, plain_tokens


def parse_condition_metadata(condition):
    """geno/cas/dox from strings like Condition_150H20S_cas-pos_dox-neg."""
    kv, plain_tokens = parse_key_value_tokens(condition)
    geno_candidates = [t for t in plain_tokens if t.lower() != "condition"]
    return {
        "geno": kv.get("geno", [geno_candidates[0] if geno_candidates else None])[0],
        "cas": kv.get("cas", [None])[0],
        "dox": kv.get("dox", [None])[0],
    }


def derive_hap40(cas, dox):
    """cas-pos AND dox-pos -> HAP40 knocked out ("neg"); either one neg -> "pos"."""
    cas_l, dox_l = str(cas).lower(), str(dox).lower()
    if cas_l not in {"pos", "neg"} or dox_l not in {"pos", "neg"}:
        return MISSING_VALUE
    return "neg" if (cas_l == "pos" and dox_l == "pos") else "pos"


def parse_file_metadata(folder, condition, filename, rules):
    """One lookup row: metadata for a single (condition, filename) pair."""
    folder_rules = rules["folders"].get(folder) or {}
    folder_meta = parse_folder_metadata(folder)
    defaults = {**(rules.get("defaults") or {}), **(folder_rules.get("defaults") or {})}
    kv, _ = parse_key_value_tokens(filename)

    row = {"condition": condition, "filename": filename}
    for key, column in FILENAME_KEY_TO_COLUMN.items():
        values = kv.get(key)
        if values:
            row[column] = values[0]
        elif column in defaults:
            row[column] = str(defaults[column])
        elif column == "replicate":
            row[column] = folder_rules.get("replicate", MISSING_VALUE)
        else:
            row[column] = MISSING_VALUE
    if not is_missing(row["location"]):
        row["location"] = row["location"].upper()

    conflicts = []
    condition_meta = parse_condition_metadata(condition)
    for column in ("geno", "cas", "dox"):
        from_condition, from_filename = condition_meta[column], row[column]
        if from_condition is None:
            continue
        if is_missing(from_filename):
            row[column] = from_condition
        elif from_filename != from_condition:
            conflicts.append(f"{column}: filename={from_filename} vs condition={from_condition}")
            if rules.get("condition_conflict_source", "filename") == "condition":
                row[column] = from_condition

    if is_missing(row["cell"]):
        row["cell"] = TYPE_TO_CELL.get(row["type"], defaults.get("cell", MISSING_VALUE))
    row["nocodazole_concentration"] = extract_nocodazole_concentration(filename)
    movie_match = MOVIE_NUMBER_RE.search(filename)
    row["movie_number"] = movie_match.group(1) if movie_match else MISSING_VALUE
    row["group"] = folder_meta["group"]
    row["experiment_date"] = folder_meta["experiment_date"]
    row["hap40"] = None

    applied_overrides = []
    for text, updates in (rules.get("manual_filename_overrides") or {}).items():
        if text in filename:
            for column, value in updates.items():
                row[column] = value
                applied_overrides.append(f"{column}={value}")
    if row["hap40"] is None:
        row["hap40"] = derive_hap40(row["cas"], row["dox"])

    row["identifier"] = create_identifier_polars(row["cell"], row["type"], row["mol"], row["geno"], row["group"])
    known_keys = set(FILENAME_KEY_TO_COLUMN) | SPECIAL_FILENAME_KEYS
    row["check_unrecognized_keys"] = ", ".join(f"{k}-{'/'.join(v)}" for k, v in kv.items() if k not in known_keys)
    row["check_duplicate_keys"] = ", ".join(f"{k}-{'/'.join(v)}" for k, v in kv.items() if len(v) > 1)
    row["check_conflicts"] = "; ".join(conflicts)
    row["check_overrides"] = ", ".join(applied_overrides)
    return row


def build_metadata_lookup(folder, pairs_df, rules):
    rows = [parse_file_metadata(folder, c, f, rules) for c, f in pairs_df.select(KEY_COLS).iter_rows()]
    schema = {col: pl.Utf8 for col in KEY_COLS + METADATA_COLUMNS + CHECK_COLUMNS}
    return pl.DataFrame(rows, schema=schema).sort(KEY_COLS)


def apply_known_corrections(df, folder, rules):
    """Fix files recorded under the wrong Condition_ folder or with a wrong token in the filename."""
    out = df.with_columns([pl.col(col).cast(pl.Utf8) for col in KEY_COLS])
    if "original_condition" not in out.columns:
        out = out.with_columns(pl.col("condition").alias("original_condition"), pl.col("filename").alias("original_filename"))
    condition_fixes = (rules.get("condition_corrections") or {}).get(folder, {})
    if condition_fixes:
        out = out.with_columns(
            pl.col("original_filename").replace_strict(condition_fixes, default=pl.col("condition"), return_dtype=pl.Utf8).alias("condition")
        )
    filename_fixes = (rules.get("filename_corrections") or {}).get(folder, {})
    if filename_fixes:
        out = out.with_columns(
            pl.col("original_filename").replace_strict(filename_fixes, default=pl.col("filename"), return_dtype=pl.Utf8).alias("filename")
        )
    return out


def annotate_folder_df(df, lookup, folder, master_dir):
    """Attach lookup metadata to one folder's dataframe without merging folders."""
    out = df.with_columns([pl.col(col).cast(pl.Utf8) for col in KEY_COLS])
    if "unique_id" in out.columns and "source_unique_id" not in out.columns:
        out = out.with_columns(pl.col("unique_id").cast(pl.Utf8).alias("source_unique_id"))
    if "window_uid" in out.columns and "source_window_uid" not in out.columns:
        out = out.with_columns(pl.col("window_uid").cast(pl.Utf8).alias("source_window_uid"))

    out = out.drop([col for col in METADATA_COLUMNS + ["foldername", "source_folder"] if col in out.columns])
    row_meta = out.select(KEY_COLS).join(lookup.select(KEY_COLS + METADATA_COLUMNS), on=KEY_COLS, how="left", maintain_order="left")
    assert row_meta.height == out.height, "Metadata join changed the row count"
    out = out.with_columns(row_meta.select(METADATA_COLUMNS).get_columns())

    folder_path = os.path.join(str(master_dir), folder)
    folder_name_with_slash = folder_path + ("" if folder_path.endswith(("\\", "/")) else "\\")
    out = out.with_columns(pl.lit(folder_name_with_slash).alias("foldername"))
    legacy_updates = [pl.col(src).alias(legacy) for legacy, src in LEGACY_COLUMN_SOURCES.items() if legacy in out.columns]
    if legacy_updates:
        out = out.with_columns(legacy_updates)
    return out


def annotate_all_folders(instant_by_folder, windowed_by_folder, rules):
    """Corrections + metadata for every folder. Returns (instant_by_folder, windowed_by_folder, lookup_by_folder)."""
    lookup_by_folder = {}
    for folder in rules["folders"]:
        tables = {name: store[folder] for name, store in (("instant", instant_by_folder), ("windowed", windowed_by_folder)) if folder in store}
        tables = {name: apply_known_corrections(df, folder, rules) for name, df in tables.items()}
        pairs = pl.concat([df.select(KEY_COLS) for df in tables.values()]).unique()
        lookup = build_metadata_lookup(folder, pairs, rules)
        lookup_by_folder[folder] = lookup
        if "instant" in tables:
            instant_by_folder[folder] = annotate_folder_df(tables["instant"], lookup, folder, rules["master_dir"])
        if "windowed" in tables:
            windowed_by_folder[folder] = annotate_folder_df(tables["windowed"], lookup, folder, rules["master_dir"])
        print(f"annotated {folder}: {lookup.height} files")
    return instant_by_folder, windowed_by_folder, lookup_by_folder


# ----------------------------------------------------------------------------- per-folder checks

def collect_folder_issues(folder, lookup, instant_df=None, windowed_df=None):
    folder_meta = parse_folder_metadata(folder)
    expected_type = PREP_TO_TYPE.get(folder_meta["folder_prep"])
    group_q = folder_meta["group_q"]
    issues = []

    def add(issue, filename="", detail=""):
        issues.append({"folder": folder, "issue": issue, "filename": filename, "detail": detail})

    for row in lookup.iter_rows(named=True):
        filename = row["filename"]
        missing = [col for col in IDENTIFIER_COLUMNS if col in row and is_missing(row[col])]
        if missing:
            add("missing_value", filename, ", ".join(missing))
        if row["check_conflicts"]:
            add("condition_vs_filename_conflict", filename, f"{row['condition']}: {row['check_conflicts']}")
        if row["check_unrecognized_keys"]:
            add("unrecognized_filename_key", filename, row["check_unrecognized_keys"])
        if row["check_duplicate_keys"]:
            add("duplicate_filename_key", filename, row["check_duplicate_keys"])
        if row["check_overrides"]:
            add("manual_override_applied", filename, row["check_overrides"])
        if expected_type and row["type"] != expected_type:
            add("type_vs_folder_mismatch", filename, f"type={row['type']}, folder is {folder_meta['folder_prep']}")
        if group_q and not is_missing(row["geno"]):
            unexpected = set(re.findall(r"(\d+)[A-Za-z]", row["geno"])) - {WILDTYPE_CAG, group_q}
            if unexpected:
                add("geno_vs_group_mismatch", filename, f"geno={row['geno']}, group={row['group']}")
        if row["hap40"] != derive_hap40(row["cas"], row["dox"]):
            add("hap40_rule_mismatch", filename, f"hap40={row['hap40']}")

    file_sets = {}
    for label, df in (("instant", instant_df), ("windowed", windowed_df)):
        if df is None:
            add("dataframe_missing", "", label)
            continue
        file_sets[label] = set(df.get_column("filename").unique().to_list())
        n_unmatched = df.filter(pl.col("location").is_null()).height
        if n_unmatched:
            add("rows_without_metadata", "", f"{label}: {n_unmatched:,} rows")
    if len(file_sets) == 2:
        for filename in sorted(file_sets["windowed"] - file_sets["instant"]):
            add("in_windowed_not_instant", filename)
        for filename in sorted(file_sets["instant"] - file_sets["windowed"]):
            add("in_instant_not_windowed (can be normal: no full windows)", filename)
    return pl.DataFrame(issues, schema=ISSUE_SCHEMA)


def summarize_issues(issues):
    """One row per (issue, detail) with the number of files and an example filename."""
    return (
        issues.group_by("issue", "detail")
        .agg(
            pl.col("filename").filter(pl.col("filename") != "").n_unique().alias("n_files"),
            pl.col("filename").sort().first().alias("example_filename"),
        )
        .sort("issue", "detail")
    )


def check_folder(folder, lookup, instant_df=None, windowed_df=None, show_issue_files=False, max_rows=300):
    """Print what each folder contributes and any problems. Read this before merging."""
    print("=" * 100)
    print(folder)
    for key, value in parse_folder_metadata(folder).items():
        print(f"  {key}: {value}")
    print(f"  movies: {lookup.get_column('filename').n_unique()}")
    combo_cols = [col for col in IDENTIFIER_COLUMNS if col != "movie_number"]
    with pl.Config(tbl_rows=max_rows, tbl_cols=40, fmt_str_lengths=120, tbl_width_chars=250):
        print(lookup.group_by(["condition"] + combo_cols).agg(pl.len().alias("n_files")).sort(["condition"] + combo_cols))
        issues = collect_folder_issues(folder, lookup, instant_df, windowed_df)
        print(f"Issues: {issues.height}")
        if issues.height:
            print(summarize_issues(issues))
            if show_issue_files:
                print(issues.sort("issue", "filename"))
    return issues


def schema_differences(dfs_by_folder, label):
    """Columns whose dtype differs between folders, or that some folders lack."""
    rows = [{"column": c, "folder": f, "dtype": str(t)} for f, df in dfs_by_folder.items() for c, t in df.schema.items()]
    if not rows:
        return None
    diffs = (
        pl.DataFrame(rows)
        .group_by("column")
        .agg(
            pl.col("dtype").n_unique().alias("n_dtypes"),
            pl.col("folder").n_unique().alias("n_folders_with_column"),
            pl.concat_str([pl.col("folder"), pl.col("dtype")], separator=": ").sort().alias("folder_dtypes"),
        )
        .filter((pl.col("n_dtypes") > 1) | (pl.col("n_folders_with_column") < len(dfs_by_folder)))
        .sort("column")
    )
    print(f"{label}: {diffs.height} columns differ between folders")
    return diffs


# ----------------------------------------------------------------------------- merge and IDs

def merge_folders(dfs_by_folder, folders):
    return pl.concat([dfs_by_folder[f] for f in folders if f in dfs_by_folder], how="diagonal_relaxed")


def normalise_genotype_names(df):
    """HTTCAG72 -> HTT72CAG in geno and genotype."""
    updates = [pl.col(c).cast(pl.Utf8).str.replace("HTTCAG72", "HTT72CAG") for c in ("genotype", "geno") if c in df.columns]
    return df.with_columns(updates) if updates else df


def fix_location_from_loc_underscore(df):
    """Old filenames start loc_GC_... (underscore, not dash)."""
    return df.with_columns(
        pl.when((pl.col("location").cast(pl.Utf8) == MISSING_VALUE) & pl.col("filename").cast(pl.Utf8).str.starts_with("loc_"))
        .then(pl.col("filename").cast(pl.Utf8).str.extract(r"^loc_([^_]+)", 1))
        .otherwise(pl.col("location").cast(pl.Utf8))
        .alias("location")
    )


def folder_basename_expr():
    return pl.col("foldername").cast(pl.Utf8).str.replace_all(r"[\\/]+$", "").str.extract(r"([^\\/]+)$", 1).fill_null("unknown-folder")


def safe_component_expr(expr, default="na"):
    cleaned = expr.cast(pl.Utf8).fill_null(default).str.replace_all(r"[^A-Za-z0-9]+", "-").str.strip_chars("-")
    return pl.when(cleaned.is_null() | (cleaned == "")).then(pl.lit(default)).otherwise(cleaned)


def column_text_expr(df, col, default="na"):
    return pl.col(col).cast(pl.Utf8).fill_null(default) if col in df.columns else pl.lit(default)


def first_alnum_expr(df, col, default="x"):
    if col in df.columns:
        return pl.col(col).cast(pl.Utf8).str.extract(r"([A-Za-z0-9])", 1).fill_null(default).str.to_lowercase()
    return pl.lit(default)


def stable_hash(value):
    return hashlib.blake2b(str(value).encode("utf-8"), digest_size=5).hexdigest()


def preserve_source_columns(df):
    out = df
    if "source_unique_id" not in out.columns and "unique_id" in out.columns:
        out = out.with_columns(pl.col("unique_id").cast(pl.Utf8).alias("source_unique_id"))
    if "source_window_uid" not in out.columns and "window_uid" in out.columns:
        out = out.with_columns(pl.col("window_uid").cast(pl.Utf8).alias("source_window_uid"))
    if "foldername" in out.columns:
        out = out.with_columns(folder_basename_expr().alias("source_folder"))
    elif "source_folder" not in out.columns:
        out = out.with_columns(pl.lit("unknown-folder").alias("source_folder"))
    return out


def rebuild_short_unique_id_polars(df):
    """uid_<location><type0><mol0><geno0><group0>_p<particle>_r<replicate>_<hash> (deterministic)."""
    out = preserve_source_columns(df)
    out = out.drop([c for c in ("_uid_key", "_uid_hash") if c in out.columns])
    particle_col = "particle" if "particle" in out.columns else "particleint" if "particleint" in out.columns else None
    particle_expr = column_text_expr(out, particle_col, "na") if particle_col else pl.lit("na")
    replicate_expr = column_text_expr(out, "replicate", "none")

    uid_key_expr = pl.concat_str(
        [column_text_expr(out, "source_folder"), column_text_expr(out, "condition"), column_text_expr(out, "filename"), particle_expr, replicate_expr],
        separator="|",
    )
    out = out.with_columns(uid_key_expr.alias("_uid_key"))
    hash_lookup = out.select("_uid_key").unique().with_columns(pl.col("_uid_key").map_elements(stable_hash, return_dtype=pl.Utf8).alias("_uid_hash"))
    out = out.join(hash_lookup, on="_uid_key", how="left", maintain_order="left")

    short_metadata_code = pl.concat_str(
        [
            safe_component_expr(column_text_expr(out, "location", "loc"), "loc"),
            first_alnum_expr(out, "type"),
            first_alnum_expr(out, "mol"),
            first_alnum_expr(out, "geno"),
            first_alnum_expr(out, "group"),
        ],
        separator="",
    )
    out = out.with_columns(
        pl.concat_str(
            [
                pl.lit("uid"),
                short_metadata_code,
                pl.lit("p") + safe_component_expr(particle_expr, "na"),
                pl.lit("r") + safe_component_expr(replicate_expr, "none"),
                pl.col("_uid_hash"),
            ],
            separator="_",
        ).alias("unique_id")
    )
    return out.drop(["_uid_key", "_uid_hash"])


def rebuild_matching_window_uid_polars(df):
    """window_uid = new unique_id + the window part (<time_window>_<start>_<end>) of the original window_uid."""
    if df is None or "unique_id" not in df.columns:
        return df
    out = preserve_source_columns(df)
    source_col = "source_window_uid" if "source_window_uid" in out.columns else "window_uid"
    out = out.with_columns(pl.col(source_col).cast(pl.Utf8).str.split("_").list.slice(2).list.join("_").alias("_window_parts"))
    no_parts = pl.col("_window_parts").is_null() | (pl.col("_window_parts") == "")
    n_missing = out.filter(no_parts).height
    if n_missing:
        print(f"Warning: {n_missing:,} rows have no window part in {source_col}; window_uid left null.")
    out = out.with_columns(
        pl.when(no_parts).then(pl.lit(None, dtype=pl.Utf8)).otherwise(pl.col("unique_id").cast(pl.Utf8) + "_" + pl.col("_window_parts")).alias("window_uid")
    )
    return out.drop("_window_parts")


def rebuild_ids(df):
    return rebuild_matching_window_uid_polars(rebuild_short_unique_id_polars(df))


# ----------------------------------------------------------------------------- verification

def expected_hap40_expr():
    cas = pl.col("cas").cast(pl.Utf8).str.to_lowercase()
    dox = pl.col("dox").cast(pl.Utf8).str.to_lowercase()
    valid = cas.is_in(["pos", "neg"]) & dox.is_in(["pos", "neg"])
    return pl.when(valid & (cas == "pos") & (dox == "pos")).then(pl.lit("neg")).when(valid).then(pl.lit("pos")).otherwise(pl.lit(MISSING_VALUE))


def verify_master_df(df, df_name, folders, expect_unique_window_rows=False):
    """Hard checks after the ID rebuild. Raises AssertionError if anything is wrong."""
    print(f"\n{'=' * 80}\n{df_name}\n{'=' * 80}")
    required = ["unique_id", "source_window_uid", "filename", "window_uid", "particle", "replicate"] + METADATA_COLUMNS
    missing_cols = [c for c in required if c not in df.columns]
    assert not missing_cols, f"{df_name}: missing columns {missing_cols}"

    df_check = df.with_columns(folder_basename_expr().alias("folder_basename"))
    found = set(df_check.get_column("folder_basename").unique().to_list())
    missing_folders = sorted(set(folders) - found)
    print(f"folders found: {len(found)}  missing: {missing_folders or 'none'}")

    bad_replicate = df_check.filter(pl.col("replicate").is_null() | pl.col("replicate").cast(pl.Utf8).is_in(["none", MISSING_VALUE])).height
    null_cols = [c for c in METADATA_COLUMNS if df_check.get_column(c).null_count() > 0]
    folder_to_group = {f: parse_folder_metadata(f)["group"] for f in folders}
    group_mismatch = df_check.filter(
        pl.col("group").cast(pl.Utf8) != pl.col("folder_basename").replace_strict(folder_to_group, default=None, return_dtype=pl.Utf8)
    ).height
    hap40_mismatch = df_check.filter(pl.col("hap40").cast(pl.Utf8) != expected_hap40_expr()).height

    source_track_key = pl.concat_str([pl.col(c).cast(pl.Utf8) for c in ("folder_basename", "condition", "filename", "particle", "replicate")], separator="|")
    collisions = (
        df_check.with_columns(source_track_key.alias("source_track_key"))
        .group_by("unique_id")
        .agg(pl.col("source_track_key").drop_nulls().n_unique().alias("n_source_tracks"))
        .filter(pl.col("n_source_tracks") > 1)
        .height
    )
    uid = pl.col("unique_id").cast(pl.Utf8)
    bad_prefix = df_check.filter(pl.col("window_uid").cast(pl.Utf8).str.slice(0, uid.str.len_chars()) != uid).height
    duplicate_windows = df_check.group_by("window_uid").len().filter(pl.col("len") > 1).height if expect_unique_window_rows else 0

    with pl.Config(tbl_rows=100, tbl_cols=20, fmt_str_lengths=60):
        print(
            df_check.group_by("folder_basename", "group", "type", "geno", "cas", "dox", "hap40", "dday", "sur")
            .agg(pl.col("unique_id").n_unique().alias("n_tracks"), pl.col("filename").n_unique().alias("n_files"), pl.len().alias("n_rows"))
            .sort("folder_basename", "geno", "cas", "dox", "dday", "sur")
        )
    print(
        f"missing replicate rows: {bad_replicate:,} | null metadata: {null_cols or 'none'} | group mismatch rows: {group_mismatch:,}\n"
        f"hap40 rule mismatch rows: {hap40_mismatch:,} | unique_id collisions: {collisions:,} | "
        f"window_uid not starting with unique_id: {bad_prefix:,} | duplicate window_uids: {duplicate_windows:,}"
    )
    assert not missing_folders, f"{df_name}: missing folders"
    assert bad_replicate == 0, f"{df_name}: rows without replicate"
    assert not null_cols, f"{df_name}: null metadata in {null_cols}"
    assert group_mismatch == 0, f"{df_name}: group does not match folder name"
    assert hap40_mismatch == 0, f"{df_name}: hap40 does not follow the cas/dox rule"
    assert collisions == 0, f"{df_name}: rebuilt unique_id collides across source tracks"
    assert bad_prefix == 0, f"{df_name}: window_uid does not start with unique_id"
    assert duplicate_windows == 0, f"{df_name}: duplicate window_uid in the window table"


def verify_instant_windowed_window_uid_mapping(instant_df, windowed_df):
    """Every good window must have frames in the instant table."""
    windowed = windowed_df.filter(pl.col("bad_fit_flag") == False) if "bad_fit_flag" in windowed_df.columns else windowed_df  # noqa: E712
    instant_wuids = instant_df.select("window_uid").drop_nulls().unique()
    windowed_wuids = windowed.select("window_uid").drop_nulls().unique()
    missing = windowed_wuids.join(instant_wuids, on="window_uid", how="anti")
    missing_tracks = windowed.select("unique_id").unique().join(instant_df.select("unique_id").unique(), on="unique_id", how="anti")
    print(f"good windows: {windowed_wuids.height:,} | missing from instant: {missing.height:,} | tracks missing from instant: {missing_tracks.height:,}")
    assert missing.height == 0, "Some window_uids are missing from the instant table"
    assert missing_tracks.height == 0, "Some unique_ids are missing from the instant table"


def save_master_tables(instant_df, windowed_df, save_dir, lookup_by_folder=None):
    os.makedirs(save_dir, exist_ok=True)
    instant_df.write_parquet(os.path.join(save_dir, "master_instant_df.parquet"))
    windowed_df.write_parquet(os.path.join(save_dir, "master_windowed_df.parquet"))
    if lookup_by_folder:
        pl.concat([lk.with_columns(pl.lit(f).alias("source_folder")) for f, lk in lookup_by_folder.items()]).write_csv(
            os.path.join(save_dir, "metadata_lookup_by_file.csv")
        )
    print(f"Saved master_instant_df.parquet and master_windowed_df.parquet to {save_dir}")


def load_master_tables(save_dir):
    return (
        pl.read_parquet(os.path.join(save_dir, "master_instant_df.parquet")),
        pl.read_parquet(os.path.join(save_dir, "master_windowed_df.parquet")),
    )
