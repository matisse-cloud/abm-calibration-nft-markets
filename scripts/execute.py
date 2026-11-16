"""Execute a notebook in place (cwd = notebooks/)."""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path(sys.argv[1]).resolve()
nb = nbformat.read(path, as_version=4)
t0 = time.time()
NotebookClient(nb, timeout=1800, kernel_name="python3",
               resources={"metadata": {"path": str(path.parent)}}).execute()
nbformat.write(nb, path)
print(f"executed {path.name} in {time.time() - t0:.0f}s")
