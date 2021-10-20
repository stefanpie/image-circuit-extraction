import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, text

import cv2

import glob
from pprint import pprint as pp

import xml.etree.ElementTree as ET


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_data = {}
    image_data["id"] = root.find('filename').text.removesuffix(".jpg")
    image_data["shape"] = [root.find('size/height').text, root.find('size/width').text, root.find('size/depth').text]
    image_data["objects"] = []


    for boxes in root.iter('object'):
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        object_name = boxes.find("name").text
        bb = [xmin, ymin, xmax, ymax]

        object_data = {"name": object_name, "bb": bb}
        image_data["objects"].append(object_data)

    return image_data


image = cv2.imread('./data/CGHD-1152/C1_D1_P1.jpg', cv2.IMREAD_COLOR)
image_data = read_content("./data/CGHD-1152/C1_D1_P1.xml")

fig, axs = plt.subplots(1,2)
axs[0].xaxis.tick_top()
axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
for obj in image_data["objects"]:
    axs[0].add_patch(patches.Rectangle((obj["bb"][0], obj["bb"][3]), obj["bb"][2] -
                 obj["bb"][0], obj["bb"][1]-obj["bb"][3], fill=False, edgecolor='red', lw=1))

obj = image_data["objects"][3]
pp(image_data)
axs[1].imshow(image[obj["bb"][1]:obj["bb"][3], obj["bb"][0]:obj["bb"][2], :])
axs[1].set_title(obj["name"])
plt.show()

#TODO work on extracting crops from images to make templates automaticly