"""Helpers to assemble curated notebooks from the original team notebooks,
keeping their saved outputs (the NFT dataset is not redistributed, so these
notebooks cannot be re-executed here)."""
import base64
import copy
import re
from pathlib import Path

import nbformat as nbf

from translations import translate

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "legacy"
LOCAL_PATH = re.compile(r"(/Users/|/var/folders/|C:\\\\Users|C:\\Users|ipykernel_\d+)")


def load(name):
    return nbf.read(LEGACY / f"{name}.ipynb", as_version=4)


def clean_outputs(outputs):
    """Drop stderr streams (warnings, progress bars, local paths) and translate text."""
    kept = []
    for o in outputs:
        if o.get("output_type") == "stream" and o.get("name") == "stderr":
            continue
        o = copy.deepcopy(o)
        if o.get("output_type") == "stream":
            o["text"] = translate(o["text"])
        elif "data" in o and "text/plain" in o["data"]:
            o["data"]["text/plain"] = translate(o["data"]["text/plain"])
        text = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if LOCAL_PATH.search(text if isinstance(text, str) else "".join(text)):
            continue
        kept.append(o)
    return kept


def code_from(nb, i, replace=None):
    """Copy code cell i with its outputs, translated and cleaned."""
    c = nb.cells[i]
    assert c.cell_type == "code", (i, c.cell_type)
    src = translate(c.source)
    for old, new in (replace or []):
        assert old in src, (i, old)
        src = src.replace(old, new)
    new = nbf.v4.new_code_cell(src.rstrip() + "\n" if src.endswith("\n\n") else src)
    new.source = new.source.rstrip()
    new.outputs = clean_outputs(c.get("outputs", []))
    new.execution_count = c.get("execution_count")
    return new


def md_from(nb, i, replace=None):
    c = nb.cells[i]
    assert c.cell_type == "markdown", (i, c.cell_type)
    src = c.source
    for old, new in (replace or []):
        assert old in src, (i, old)
        src = src.replace(old, new)
    return nbf.v4.new_markdown_cell(src.strip())


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def save_png(cell, path, index=0):
    """Write the index-th PNG output of a cell to path."""
    pngs = [o["data"]["image/png"] for o in cell.outputs if "image/png" in o.get("data", {})]
    Path(path).write_bytes(base64.b64decode(pngs[index]))


def write(cells, out):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbf.validate(nb)
    nbf.write(nb, out)
    print("wrote", out)
