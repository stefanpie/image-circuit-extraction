import numpy as np
import imutils
import glob
import cv2
from skimage.feature import peak_local_max
from skimage.feature import match_template
import matplotlib.pyplot as plt
from scipy import stats



visualize = True

template = cv2.imread("./data_old/components/resistor/0.png")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# template = cv2.Canny(template, 50, 200)


image = cv2.imread("./data_old/circuits/0.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 50, 200)
# cv2.imshow("image", image)
# cv2.waitKey(0)
# # loop over the scales of the image


for i, scale in enumerate(np.linspace(0.5, 0.8, 5)):
    for j, rotate in enumerate(np.linspace(0, 180, 3)):
        print(f"scale: {scale}, rotate: {rotate}")
        template_rotated = imutils.rotate_bound(template, rotate)
        template_resized = imutils.resize(template_rotated, width=int(template.shape[1] * scale))
        corr = match_template(image, template_resized)
        coordinates = peak_local_max(corr, min_distance=np.rint(
            np.min(template_resized.shape)*0.9).astype(), exclude_border=20, p_norm=3)
        # print(np.min(corr), np.max(corr))
        
        fig, axs = plt.subplots(2, 1)
        axs[1].imshow(corr)
        axs[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
        
        axs[0].imshow(image)
        axs[0].plot(coordinates[:, 1], coordinates[:, 0], 'r.')

        for x, y in zip(coordinates[:, 1], coordinates[:, 0]):
            axs[0].add_patch(plt.Rectangle((x, y),
                                           template_resized.shape[1], template_resized.shape[0],
                                           edgecolor='r', facecolor='none'))
        plt.show()
