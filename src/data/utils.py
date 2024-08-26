import json
from pathlib import Path
from typing import Dict, List


# Checks if a bounding box in xyxy format is contained within another
def is_bbox_a_in_bbox_b(a: List[float], b: List[float]) -> bool:
    return a[0] >= b[0] and a[1] >= b[1] and a[2] <= b[2] and a[3] <= b[3]


def write_json(content: Dict, path: Path):
    with path.open("w") as file:
        json.dump(content, file)
