import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eksperimen_model.m4human_training import extract_cli

if __name__ == "__main__":
    extract_cli('U')
