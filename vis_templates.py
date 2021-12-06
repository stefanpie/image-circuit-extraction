import numpy as np
import matplotlib.pyplot as plt

import cv2

import glob
from pprint import pprint as pp
from pathlib import Path

import tqdm
import imagesize




labels = glob.glob("./data/templates/*")
labels = [Path(l).stem for l in labels]
labels = sorted(set(labels))

pp(labels)

ignore_labels = ["text", "junction", "corssover", "terminal"]
labels = [l for l in labels if l not in ignore_labels]



shapes = []
label_idx = []

for l in labels:
    pp(f"/data/templates/{l}/*")
    label_template_files = glob.glob(f"./data/templates/{l}/*")
    for fp in tqdm.tqdm(label_template_files):
        shapes.append(imagesize.get(fp))
        label_idx.append(labels.index(l))

fig, ax = plt.subplots(1,1)
# pp(label_idx)

plt.scatter([s[1] for s in shapes], [s[0] for s in shapes], c=label_idx,s=0.5)

# This function formatter will replace integers with target names
formatter = plt.FuncFormatter(lambda val, loc: labels[val.astype(int)])

plt.colorbar(ticks=range(len(labels)), format=formatter)
plt.show()


