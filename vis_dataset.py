import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, text

import cv2

import glob
from pprint import pprint as pp

from extract_templates import read_content


image = cv2.imread('./data/CGHD-1152/C1_D1_P1.jpg', cv2.IMREAD_COLOR)
image_data = read_content("./data/CGHD-1152/C1_D1_P1.xml")

fig, axs = plt.subplots(1,1)
axs.xaxis.tick_top()
axs.imshow(cv2.cvtColor(image[:800,:], cv2.COLOR_BGR2RGB))
for obj in image_data["objects"]:
    axs.add_patch(patches.Rectangle((obj["bb"][0], obj["bb"][3]), obj["bb"][2] -
                 obj["bb"][0], obj["bb"][1]-obj["bb"][3], fill=False, edgecolor='red', lw=1))
    axs.text(obj["bb"][0], obj["bb"][3]+8,  obj["name"], fontsize=10)

# obj = image_data["objects"][1]
# pp(image_data)
# axs[1].imshow(image[obj["bb"][1]:obj["bb"][3], obj["bb"][0]:obj["bb"][2], :])
# axs[1].set_title(obj["name"])
plt.show()
