import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats

from extract_templates import read_content

import glob
from pathlib import Path
from pprint import pprint as pp
import tqdm
import random
import pickle
import joblib

from sklearn.linear_model import LogisticRegression, RidgeClassifier, RANSACRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

from skimage.feature import shape_index
from skimage.measure import moments_hu, moments_normalized, moments_coords_central

random.seed(0)

image_files = glob.glob(f"./data/templates/**/*.jpg")
image_files = sorted(image_files)
image_files = [Path(t) for t in image_files]

include_labels = ["resistor", "capacitor_polarized",
                  "capacitor_unpolarized", "diode", "gnd", "voltage_dc", "voltage_dc_ac", "inductor"]

image_files = [t for t in image_files if Path(t).parent.stem in include_labels]

image_files_sub = random.sample(image_files, 6000)

labels = [Path(t).parent.stem for t in image_files_sub]

for i, l in enumerate(labels):
    if "capacitor" in l:
        labels[i] = "capacitor"


def load_image(img_fp):
    image = cv2.imread(str(img_fp))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = image/255.0
    image = 255-cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 25, 0)
    return image


images = list(map(load_image, tqdm.tqdm(image_files_sub, total=len(image_files_sub))))


def img_feature_extraction(img):
    moments = cv2.moments(img, )
    moments_array = np.array(list(moments.values()))
    huMoments = cv2.HuMoments(moments)
    for i in range(0, 7):
        huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
    huMoments = np.ravel(huMoments)
    features = np.concatenate((moments_array, huMoments))
    return features


img_features = list(map(img_feature_extraction, tqdm.tqdm(images, total=len(images))))
x = np.array(img_features)

y = np.array(labels)

pipe = Pipeline([('scaler', RobustScaler()),
                 ('model', RandomForestClassifier(verbose=11, n_estimators=200))])

# le = LabelEncoder()
# le.fit(y)
# y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# model = RandomForestClassifier(verbose=11, n_estimators=100)
# model = MLPClassifier(hidden_layer_sizes=(500,400,300,200), verbose=11, max_iter=1000)

model = pipe

model.fit(x_train, y_train)
score = model.score(x_test, y_test)
print()
print(score)

predictions = model.predict(x_test)
print(predictions)


joblib.dump(pipe, "./model.sklearn")

cm = confusion_matrix(y_test, predictions, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

plt.show()
