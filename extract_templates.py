import xml.etree.ElementTree as ET
from pathlib import Path
import glob
from pprint import pprint as pp
import os
import cv2

import tqdm


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_data = {}
    image_data["id"] = root.find('filename').text.replace(".jpg", "")
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


DATA_DIR = Path("./data")

files_xml = glob.glob(str(DATA_DIR)+"/CGHD-1152/*.xml")
files_xml = sorted(files_xml)
files_jpg = [x.replace(".xml", ".jpg") for x in files_xml]

labels = set()
for fp_xml in files_xml:
    image_data = read_content(fp_xml)
    for obj in image_data["objects"]:
        labels.add(obj["name"])

os.makedirs(DATA_DIR/"templates", exist_ok=True)
for label in labels:
    os.makedirs(DATA_DIR/"templates"/label, exist_ok=True)

for fp_xml, fp_jpg in tqdm.tqdm(list(zip(files_xml, files_jpg))):
    image = cv2.imread(fp_jpg, cv2.IMREAD_COLOR)
    image_data = read_content(fp_xml)
    img_id = image_data["id"]
    for obj in image_data["objects"]:
        label = obj["name"]
        bb = obj["bb"]
        file_name = f"{img_id}_{label}_{'_'.join(map(str, bb))}.jpg"
        img_crop = image[obj["bb"][1]:obj["bb"][3], obj["bb"][0]:obj["bb"][2], :]
        # print(DATA_DIR/"templates"/label/file_name)
        cv2.imwrite(str(DATA_DIR/"templates"/label/file_name), img_crop)
