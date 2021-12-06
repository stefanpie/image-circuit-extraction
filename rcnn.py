# import pandas as pd
# import numpy as np
# import cv2
# import torch
# import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection import FasterRCNN
# from torch.utils.data import DataLoader, Dataset

# import os
# import glob
# import xml.etree.ElementTree as ET


# class CircuitDataset(Dataset):
#     def __init__(self, data_dir) -> None:
#         super().__init__()
#         self.data_dir = data_dir

#         self.label_files = glob.glob(f"{data_dir}/*.xml")
#         self.label_files = sorted(self.label_files)

#         self.image_files = [x.replace(".xml", ".jpg") for x in self.label_files]

#         self.labels = self.get_all_labels(self.label_files)
#         self.labels = sorted(self.labels)

#         self.image_transform = torchvision.transforms.ToTensor()

#     def __getitem__(self, index: int):
#         label_file = self.label_files[index]
#         image_file = self.image_files[index]

#         image = cv2.imread(image_file)
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
#         image = image / 255.0
#         image = self.image_transform(image)

#         label_data = self.read_label_file(label_file)

#         boxes = []
#         labels = []
#         for obj in label_data["objects"]:
#             boxes.append(obj["bb"])
#             labels.append(obj["name"])

#         labels = [self.labels.index(l)+1 for l in labels]

#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)

#         target = {'boxes': boxes, "labels": labels}
#         return image, target

#     def __len__(self) -> int:
#         return len(self.label_files)

#     def read_label_file(self, xml_file: str):

#         tree = ET.parse(xml_file)
#         root = tree.getroot()

#         image_data = {}
#         image_data["id"] = root.find('filename').text.replace(".jpg", "")
#         image_data["shape"] = [root.find('size/height').text, root.find('size/width').text,
#                                root.find('size/depth').text]
#         image_data["objects"] = []

#         for boxes in root.iter('object'):
#             ymin = int(boxes.find("bndbox/ymin").text)
#             xmin = int(boxes.find("bndbox/xmin").text)
#             ymax = int(boxes.find("bndbox/ymax").text)
#             xmax = int(boxes.find("bndbox/xmax").text)
#             object_name = boxes.find("name").text.replace(".", "_")
#             bb = [xmin, ymin, xmax, ymax]

#             object_data = {"name": object_name, "bb": bb}
#             image_data["objects"].append(object_data)

#         return image_data

#     def get_all_labels(self, xml_files: list) -> set:
#         labels = set()
#         for fp_xml in xml_files:
#             image_data = self.read_label_file(fp_xml)
#             for obj in image_data["objects"]:
#                 labels.add(obj["name"])
#         labels = list(labels)
#         return labels


# c_dataset = CircuitDataset('./data/CGHD-1152')


# def collate_fn(batch):
#     return tuple(zip(*batch))


# train_data_loader = DataLoader(
#     c_dataset,
#     # batch_size=10,
#     # shuffle=True,
#     # num_workers=2,
#     collate_fn=collate_fn
# )

# images, targets = next(iter(train_data_loader))
# print(images)
# print(targets)


# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
# num_classes = len(c_dataset.labels)+1
# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


# params = [p for p in model.parameters() if p.requires_grad]
# optimizer = torch.optim.SGD(params, lr=0.005,
#                             momentum=0.9, weight_decay=0.0005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                step_size=3,
#                                                gamma=0.1)

# num_epochs = 31

# for epoch in range(num_epochs):
#     # train for one epoch, printing every 10 iterations
#     # Engine.pyTrain_ofOne_The epoch function takes both images and targets. to(device)
#     train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)

#     # update the learning rate
#     lr_scheduler.step()

#     # evaluate on the test dataset
#     evaluate(model, data_loader_test, device=device)

#     print('')
#     print('==================================================')
#     print('')


from detecto.core import Dataset
from detecto import core, utils
from torchvision import transforms
import matplotlib.pyplot as plt


custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(400),
    transforms.ToTensor(),
    utils.normalize_transform(),
])
# If your images and labels are in the same folder
dataset = Dataset('./data/CGHD-1152/', transform=custom_transforms)

loader = core.DataLoader(dataset, batch_size=16, shuffle=False)
labels = ["resistor", "capacitor.polarized", "capacitor.unpolarized", "diode", "gnd", "voltage.dc"]
model = core.Model()
losses = model.fit(loader, epochs=20, learning_rate=0.001, verbose=True)
plt.plot(losses)
plt.show()
