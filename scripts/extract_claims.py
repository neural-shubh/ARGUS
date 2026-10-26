import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from argus.extraction import extract_pending
from argus.storage.database import ArgusDatabase

extract_pending(ArgusDatabase())