import csv
import io
import json
import random
import textwrap
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, TypedDict   # ← IMPORTANT


from mcp.server.fastmcp import FastMCP

server = FastMCP("bishal-mcp-playground")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


def _safe_filename(name: str) -> Path:
    if "/" in name or "\\" in name:
        raise ValueError("Name must not contain path separators.")
    return DATA_DIR / name


# @server.tool()
# def greet(name: str) -> str:
#     return f"Hi {name}, this response is coming from Bishal's MCP server."


# @server.tool()
# def add(a: float, b: float) -> float:
#     return a + b


# @server.tool()
# def current_time() -> str:

#     now = datetime.now()
#     return now.strftime("%Y-%m-%d %H:%M:%S")


@server.tool()
def write_note(filename: str, text: str) -> str:

    file_path = _safe_filename(filename)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    return f"Appended note to {filename}."


@server.tool()
def list_data_files() -> List[str]:

    files: List[str] = []
    for p in DATA_DIR.iterdir():
        if p.is_file():
            files.append(p.name)
    return files


class FileInfo(TypedDict):
    name: str
    size_bytes: int
    modified: str


@server.tool()
def get_file_info(filename: str) -> FileInfo:

    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")

    stat = file_path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return {
        "name": file_path.name,
        "size_bytes": stat.st_size,
        "modified": modified,
    }


@server.tool()
def read_text_file(filename: str) -> str:

    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        raise ValueError("This file is not a readable text file.")

@server.tool()
def delete_file(filename: str) -> str:

    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")
    file_path.unlink()
    return f"Deleted file {filename}."


@server.tool()
def create_folder(folder_name: str) -> str:

    # no nested paths allowed
    folder_path = _safe_filename(folder_name)
    folder_path.mkdir(exist_ok=True)
    return f"Folder '{folder_name}' created (or already existed)."


@server.tool()
def list_folders() -> List[str]:

    folders: List[str] = []
    for p in DATA_DIR.iterdir():
        if p.is_dir():
            folders.append(p.name)
    return folders


class MatchResult(TypedDict):
    line_number: int
    line_text: str


@server.tool()
def search_in_file(filename: str, query: str) -> List[MatchResult]:

    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")

    results: List[MatchResult] = []
    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if query in line:
                results.append({
                    "line_number": i,
                    "line_text": line.rstrip("\n"),
                })
    return results


@server.tool()
def replace_in_file(filename: str, old: str, new: str, max_replacements: int = 0) -> int:

    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")

    text: str
    with file_path.open("r", encoding="utf-8") as f:
        text = f.read()

    if max_replacements and max_replacements > 0:
        new_text, count = text.replace(old, new, max_replacements), text.count(old)
        # but .replace doesn't return count when limit is set; handle manually:
        # simplest: do a loop or use split-join; to keep it simple, ignore exact count
        # and recompute via difference
        replaced_text = text.replace(old, new, max_replacements)
        count = text.count(old) - replaced_text.count(old)
        text = replaced_text
    else:
        count = text.count(old)
        text = text.replace(old, new)

    with file_path.open("w", encoding="utf-8") as f:
        f.write(text)

    return count


@server.tool()
def read_json_file(filename: str) -> Any:

    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


class CsvSummary(TypedDict):
    filename: str
    columns: List[str]
    row_count: int
    column_count: int
    sample_rows: List[Dict[str, str]]


@server.tool()
def csv_summary(filename: str, sample_size: int = 3) -> CsvSummary:

    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")

    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows: List[Dict[str, str]] = list(reader)

    sample_rows = rows[:sample_size]

    return {
        "filename": filename,
        "columns": columns,
        "row_count": len(rows),
        "column_count": len(columns),
        "sample_rows": sample_rows,
    }


class GlobalSearchHit(TypedDict):
    filename: str
    line_number: int
    line_text: str

@server.tool()
def search_in_all_files(query: str) -> List[GlobalSearchHit]:
    """
    Search for a text query across ALL text files in the 'data' folder.
    Returns a list of matches: filename, line number, and line text.
    """
    hits: List[GlobalSearchHit] = []

    for p in DATA_DIR.iterdir():
        if not p.is_file():
            continue

        # Try to read as UTF-8 text; skip binary files
        try:
            with p.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    if query in line:
                        hits.append({
                            "filename": p.name,
                            "line_number": i,
                            "line_text": line.rstrip("\n"),
                        })
        except UnicodeDecodeError:
            # skip non-text files
            continue

    return hits

@server.tool()
def summarize_text_file(filename: str, max_chars: int = 400) -> str:
    """
    Create a simple summary of a text file in the 'data' folder.
    For now this is heuristic: take the first few lines and trim to max_chars.
    """
    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        raise ValueError("This file is not a readable text file.")

    # Normalise whitespace a bit
    content = content.strip()

    if not content:
        return "File is empty."

    if len(content) <= max_chars:
        return content

    # Simple truncation with ellipsis
    return content[: max_chars - 3] + "..."


# =========================
#   PYTHON EXEC (SAFE-ish)
# =========================

@server.tool()
def run_python_snippet(code: str) -> str:
    """
    Execute a small Python snippet in a very limited environment.
    Intended for quick math or string experiments, not heavy code.

    Example:
    `2 + 3 * 4`
    `sum(range(10))`
    """
    # VERY restricted globals: basic builtins only
    allowed_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "range": range,
    }

    env: Dict[str, Any] = {}
    try:
        result = eval(code, {"__builtins__": allowed_builtins}, env)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
    return repr(result)


#Mini ML Assistant to read csv files with Pandas
# class DatasetDescription(TypedDict):
#     filename: str
#     rows: int
#     columns: List[str]
#     dtypes: Dict[str, str]
#     missing_values: Dict[str, int]


# class DatasetPreview(TypedDict):
#     filename: str
#     rows: List[Dict[str, Any]]

# @server.tool()
# def describe_dataset(filename: str) -> DatasetDescription:
#     """
#     Load a CSV file from the 'data' folder and return:
#     - number of rows
#     - column names
#     - column dtypes (as strings)
#     - missing value counts per column
#     """
#     import pandas as pd

#     file_path = _safe_filename(filename)
#     if not file_path.exists():
#         raise FileNotFoundError(f"File '{filename}' does not exist in data/")

#     df = pd.read_csv(file_path)

#     columns = list(df.columns)
#     dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
#     missing = {col: int(count) for col, count in df.isna().sum().items()}

#     return {
#         "filename": filename,
#         "rows": int(len(df)),
#         "columns": columns,
#         "dtypes": dtypes,
#         "missing_values": missing,
#     }


# @server.tool()
# def preview_dataset(filename: str, n: int = 5) -> DatasetPreview:
#     """
#     Return the first `n` rows of a CSV file as a list of dicts.
#     Good for quickly looking at the data structure.
#     """
#     import pandas as pd

#     file_path = _safe_filename(filename)
#     if not file_path.exists():
#         raise FileNotFoundError(f"File '{filename}' does not exist in data/")

#     df = pd.read_csv(file_path)
#     n = max(1, n)
#     head_df = df.head(n)

#     # Convert to plain Python types (no numpy types)
#     rows: List[Dict[str, Any]] = []
#     for _, row in head_df.iterrows():
#         rows.append({col: (None if pd.isna(val) else val) for col, val in row.items()})

#     return {"filename": filename, "rows": rows}



# class ColumnStats(TypedDict):
#     mean: float
#     std: float
#     min: float
#     max: float


# class NumericSummary(TypedDict):
#     filename: str
#     stats: Dict[str, ColumnStats]


# @server.tool()
# def numeric_summary(filename: str) -> NumericSummary:
#     """
#     For all numeric columns in a CSV file, return basic statistics:
#     mean, std, min, max.
#     """
#     import pandas as pd

#     file_path = _safe_filename(filename)
#     if not file_path.exists():
#         raise FileNotFoundError(f"File '{filename}' does not exist in data/")

#     df = pd.read_csv(file_path)
#     num_df = df.select_dtypes(include="number")

#     stats: Dict[str, ColumnStats] = {}
#     for col in num_df.columns:
#         series = num_df[col].dropna()
#         if series.empty:
#             continue

#         stats[col] = {
#             "mean": float(series.mean()),
#             "std": float(series.std(ddof=0)),
#             "min": float(series.min()),
#             "max": float(series.max()),
#         }

#     return {
#         "filename": filename,
#         "stats": stats,
#     }


#Mini Assistant without ML
class DatasetDescription(TypedDict):
    filename: str
    rows: int
    columns: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]


class DatasetPreview(TypedDict):
    filename: str
    rows: List[Dict[str, Any]]

@server.tool()
def describe_dataset(filename: str) -> DatasetDescription:
    """
    Simple CSV description without pandas:
    counts rows, columns, and missing values using the csv module.
    Dtypes are reported as 'unknown' for now.
    """
    import csv

    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")

    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return {
                "filename": filename,
                "rows": 0,
                "columns": [],
                "dtypes": {},
                "missing_values": {},
            }

        columns = header
        missing = {col: 0 for col in columns}
        row_count = 0

        for row in reader:
            row_count += 1
            for col, value in zip(columns, row):
                if value is None or value == "":
                    missing[col] += 1

    dtypes = {col: "unknown" for col in columns}

    return {
        "filename": filename,
        "rows": row_count,
        "columns": columns,
        "dtypes": dtypes,
        "missing_values": missing,
    }

@server.tool()
def preview_dataset(filename: str, n: int = 5) -> DatasetPreview:
    """
    Return the first `n` rows of a CSV file as a list of dicts.
    Implemented with the csv module (no pandas).
    """
    import csv

    file_path = _safe_filename(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{filename}' does not exist in data/")

    n = max(1, n)
    rows: List[Dict[str, Any]] = []

    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            # row is already a dict[str, str | None]
            rows.append(dict(row))

    return {"filename": filename, "rows": rows}




def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
