import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats

from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import closing, binary_opening, disk, skeletonize, square
from skimage.morphology.convex_hull import convex_hull_object


from extract_templates import read_content



visualize = True
image = cv2.imread("./data/CGHD-1152/C10_D1_P1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = 255-cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 99, 20)
# image = cv2.Canny(image, 50, 200)
# cv2.imshow("image", image)
# cv2.waitKey(0)

image = image / 255.0

image_data = read_content("./data/CGHD-1152/C10_D1_P1.xml")

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].set_title('Binarized Image', fontsize=15)
ax[0].imshow(image, cmap='gray')

# blobs = closing(image, disk(10))

blobs = closing(image, square(20))
ax[1].imshow(blobs, cmap='gray')

plt.tight_layout()
plt.show()