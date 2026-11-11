"""Download a small MathVision-Wild slice for local evaluation, not the 13.6 GB full set."""
from pathlib import Path

from datasets import load_dataset

destination = Path(__file__).resolve().parent.parent / "input" / "mathvision_wild"
destination.mkdir(parents=True, exist_ok=True)
dataset = load_dataset("MathLLMs/MathVision-Wild", split="test_photo", streaming=True)
for index, row in enumerate(dataset.take(20), 1):
    row["image"].convert("RGB").save(destination / f"{row['id']}.jpg", quality=92)
    print(f"[{index}/20] {row['id']}.jpg")

