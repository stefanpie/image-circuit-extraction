from re import match
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
import tqdm
from joblib import Parallel, delayed
from sklearn.cluster import OPTICS, DBSCAN


import cv2
import imutils
from skimage.feature import peak_local_max
from skimage.feature import match_template
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import glob
from pprint import pprint as pp
from pathlib import Path
from functools import partial
import itertools


from extract_templates import read_content


def SSIM(img1, img2):
    mssim = structural_similarity(img1, img2)
    return mssim


def PSNR(img1, img2):
    psnr = peak_signal_noise_ratio(img1, img2)
    return psnr


def MSE(img1, img2):
    mse = np.mean((img1 - img2)**2, axis=(-1, -2))
    return mse


def MAE(img1, img2):
    mae = np.mean(np.abs(img1 - img2), axis=(-1,-2))
    return mae


def rolling_window(a, shape):  # rolling window for 2D array
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)


def template_correlation(image, template, metric=MSE, scale=0.5):
    # image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)
    # template = cv2.resize(image, (int(template.shape[1] * scale), int(template.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(image)
    # axs[1].imshow(template)
    # plt.show()

    # print(image.shape)
    # print(template.shape)
    img_rolling = rolling_window(image, template.shape)
    corr = metric(img_rolling, template)
    # print(corr.shape)

    # corr = np.zeros((corr_rows, corr_cols))
    # coords = []
    # for i in tqdm.tqdm(range(corr_rows)):
    #     for j in range(corr_cols):
    #         coords.append([i, j])
    #         image_crop = image[i:i+template_rows, j:j+template_cols]
    #         # print(image_crop.shape)
    #         # print(template.shape)
    #         # print()

    #         # input("...")

    #         value = metric(image_crop, template)
    #         # print(value)
    #         corr[i,j] = value
    # def get_metric_value_at_location(loc):
    #     image_crop = image[loc[0]:loc[0]+template_rows, loc[1]:loc[1]+template_cols]
    #     value = metric(image_crop, template)
    #     return value

    # values = Parallel(n_jobs=-1, verbose=1)(delayed(get_metric_value_at_location)(loc) for loc in coords)
    # for loc, value in zip(coords, values):
    #     corr[loc[0], loc[1]] = value

    # plt.imshow(corr)
    # plt.show()
    return corr


def get_matches_single_template(image, template):
    matches = []
    corr_total = np.zeros_like(image).astype(float)
    for i, scale in enumerate(np.linspace(0.8, 1.5, 8)):
        for j, rotate in enumerate(range(0,4)):
            # print(f"scale: {scale}, rotate: {rotate}")

            template_rotated = np.rot90(template, k=rotate)
            template_resized = cv2.resize(template_rotated, (int(
                template_rotated.shape[1] * scale), int(template_rotated.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
            # template_resized = imutils.resize(template_rotated, width=int(template.shape[1] * scale))
            # print("starting corr")
            # corr = match_template(image, template_resized)
            corr = cv2.matchTemplate(image, template_resized, cv2.TM_CCOEFF_NORMED)

            # corr = (corr-min(corr))/(max(corr)-min(corr))

            # corr = template_correlation(image, template_resized)
            corr_reshape = cv2.resize(corr, (image.shape[1], image.shape[0]))
            corr_total += corr_reshape
            
            # print("starting peak finding")
            coordinates = peak_local_max(corr, min_distance=np.rint(
                np.min(template_resized.shape)*0.9).astype(int), exclude_border=20, threshold_abs=0.5)

            # print("collecting matches")
            for x, y in zip(coordinates[:, 1], coordinates[:, 0]):
                match = {}
                match["scale"] = scale
                match["rotate"] = rotate
                match["bb"] = [x, y, x+template_resized.shape[1], y+template_resized.shape[0]]
                matches.append(match)
    # plt.imshow(corr_total)
    # plt.show()
    # if visualize:
    #     fig, ax = plt.subplots(1, 1)
    #     ax.imshow(image)
    #     ax.plot([m['bb'][0] for m in matches], [m['bb'][1] for m in matches], 'r.')
    #     for m in matches:
    #         ax.add_patch(plt.Rectangle((m['bb'][0], m['bb'][1]),
    #                                     m['bb'][2]-m['bb'][0], m['bb'][3]-m['bb'][1],
    #                                     edgecolor='r', facecolor='none'))
    #     plt.show()
    return matches


def get_matches_batch_of_templates(image, templates):
    matches_grouped = Parallel(n_jobs=-1, verbose=11)(delayed(get_matches_single_template)(image, t) for t in templates)
    matches = list(itertools.chain.from_iterable(matches_grouped))
    # matches = []
    # for t in tqdm.tqdm(templates):
    #     matches_single = get_matches_single_template(image, t)
    #     matches += matches_single
    if visualize:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image)
        ax.plot([m['bb'][0] for m in matches], [m['bb'][1] for m in matches], 'r.')
        for m in matches:
            ax.add_patch(plt.Rectangle((m['bb'][0], m['bb'][1]),
                                       m['bb'][2]-m['bb'][0], m['bb'][3]-m['bb'][1],
                                       edgecolor='r', facecolor='none'))
        plt.show()
    return matches

# Malisiewicz et al.


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


visualize = True
image = cv2.imread("./data/CGHD-1152/C1_D1_P1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = 255-cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 20)
# image = cv2.Canny(image, 50, 200)
# cv2.imshow("image", image)
# cv2.waitKey(0)

image_data = read_content("./data/CGHD-1152/C1_D1_P1.xml")


template_type = "diode"
template_files = glob.glob(f"./data/templates/{template_type}/*.jpg")
template_files = sorted(template_files)
# pp(template_files)

template_0 = cv2.imread(template_files[0])
template_0 = cv2.cvtColor(template_0, cv2.COLOR_BGR2GRAY)
template_0 = 255-cv2.adaptiveThreshold(template_0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 20)
# template_0 = cv2.Canny(template_0, 50, 200)

# cv2.imshow("template_0", template_0)
# cv2.waitKey(0)

templates_batch = [cv2.imread(t_fp) for t_fp in template_files[:30]]
templates_batch = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in templates_batch]
templates_batch = [255-cv2.adaptiveThreshold(t, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 25, 0) for t in templates_batch]
# templates_batch = [cv2.Canny(t, 50, 200) for t in templates_batch]


# cv2.imshow("template_0", template_0)
# cv2.waitKey(0)

# template_correlation(image, template_0)

# matches = get_matches_single_template(image, template_0)
matches = get_matches_batch_of_templates(image, templates_batch)

bbs = [m["bb"] for m in matches]
bbs = np.array(bbs)

bbs_after = non_max_suppression_fast(bbs, 0.5)

x = [bb[0] for bb in bbs_after]
y = [bb[1] for bb in bbs_after]

fig, ax = plt.subplots(1, 1)
ax.imshow(image)
ax.plot(x, y, 'r.')
for bb in bbs_after:
    ax.add_patch(plt.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
                               edgecolor='r', facecolor='none'))
# plt.scatter(x, y)
plt.show()

# bbs = [m["bb"] for m in matches]
# x = [bb[0] for bb in bbs]
# y = [bb[1] for bb in bbs]

# matched_locs = np.array(list(zip(x, y)))
# clusterer = DBSCAN(min_samples=int(0.75*len(templates_batch)), eps=10)
# labels = clusterer.fit_predict(matched_locs)
# num_labels = len(set(labels))
# colors = [cm.viridis(float(i)/num_labels) for i in labels]
# pp(labels)
# plt.imshow(image, cmap="gray")
# plt.scatter(x, y, c=colors, s=10)
# plt.show()

# pp(matches)


print("done")
