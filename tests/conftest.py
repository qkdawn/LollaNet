from __future__ import annotations

import os


# Load PyArrow's native DLLs before Torch and Scikit-learn during test collection.
# Delayed loading after the other scientific runtimes can access-violate on Windows.
if os.name == "nt":
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pass
