import json
from pathlib import Path
filename = "data/base_set/validation/val_labels.json"

with open(filename) as json_file:
    d = json.load(json_file)

print(d)