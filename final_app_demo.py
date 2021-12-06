from matplotlib.widgets import RectangleSelector, TextBox
from matplotlib.widgets import Button
from matplotlib import patches
import matplotlib.pyplot as plt

import numpy as np
import joblib

import pprint

import cv2



current_box = None
all_boxes = []


def line_select_callback(eclick, erelease):
    'eclick and erelease are the press and release events'
    global current_box
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    current_box = [round(x1), round(y1), round(x2), round(y2)]
    image_crop = image[current_box[1]:current_box[3], current_box[0]:current_box[2], :]

    ax_img_crop.imshow(image_crop)
    fig.canvas.draw()
    fig.canvas.flush_events()
    # print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
    # print(" The button you used were: %s %s" % (eclick.button, erelease.button))


def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)


def set_box(event):
    global all_boxes
    data = {}
    data["name"] = "unknown"
    data["bb"] = current_box
    
    ax_img.add_patch(patches.Rectangle((current_box[0], current_box[3]), current_box[2] -
                                       current_box[0], current_box[1]-current_box[3], fill=False, edgecolor='red', lw=1))
    text_label = ax_img.text(current_box[0], current_box[3]+20,  data["name"], fontsize=10)
    data["text_label"] = text_label
    all_boxes.append(data)
    print(current_box)

    box_data = pprint.pformat(all_boxes, indent=4)
    text_box.set_val(box_data)

def run_ml(event):
    for b in all_boxes:
        image_crop = image[b["bb"][1]:b["bb"][3], b["bb"][0]:b["bb"][2], :]
        image_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        image_crop = 255-cv2.adaptiveThreshold(image_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 0)
        moments = cv2.moments(image_crop)
        moments_array = np.array(list(moments.values()))
        huMoments = cv2.HuMoments(moments)
        for i in range(0, 7):
            huMoments[i] = -1 * np.copysign(1.0, huMoments[i]) * np.log10(abs(huMoments[i]))
        huMoments = np.ravel(huMoments)
        features = np.concatenate((moments_array, huMoments))
        output = model.predict(features.reshape(1, -1))[0]
        print(output)

        b["name"] = output
        b["text_label"].set_text(output)


image_fp = "./simple_test_images/01.png"
image = cv2.imread(str(image_fp))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

model = joblib.load("./model.sklearn")


fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(2, 5, height_ratios=[8,2])
ax_img = fig.add_subplot(gs[0, :2])
ax_img.set_title('gs[0, :]')
ax_img.imshow(image)


ax_img_crop = fig.add_subplot(gs[0, 2:3])

ax_text_box = fig.add_subplot(gs[0, 3:4])
text_box = TextBox(ax_text_box, '')



print("\n      click  -->  release")

# drawtype is 'box' or 'line' or 'none'
toggle_selector.RS = RectangleSelector(ax_img, line_select_callback,
                                       drawtype='box', useblit=True,
                                       button=[1, 3],  # don't use middle button
                                       minspanx=5, minspany=5,
                                       spancoords='pixels',
                                       interactive=True)

plt.connect('key_press_event', toggle_selector)

ax_set_box = fig.add_subplot(gs[1, 0])
b_set_box = Button(ax_set_box, 'Set Box')
b_set_box.on_clicked(set_box)

ax_run_ml = fig.add_subplot(gs[1, 1])
b_run_ml = Button(ax_run_ml, 'Run ML')
b_run_ml.on_clicked(run_ml)

plt.show()
