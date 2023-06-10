import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import measure, color, io

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.filters import sobel
from tqdm import tqdm
DEBUG=True

def find_masks_watershed(mask, separate=True):
    # Code stolen from https://github.com/bnsreenu/python_for_microscopists/blob/master/205_predict_unet_with_watershed_single_image.py
    mask_rgb = np.stack([mask, mask, mask], -1)
    ret1, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3, 3),np.uint8)

    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=10)
    # opening = sure_bg = mask

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 0)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.15 * dist_transform.max(), 255,0)
    sure_fg = np.uint8(sure_fg)
    
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers+10
    markers[unknown == 255] = 0
    cv2.imwrite("aa.png", opening)
    cv2.imwrite("aasure_fg.png", sure_fg)
    cv2.imwrite("aamarker.png", markers)
    labels = cv2.watershed(mask_rgb, markers)
    # labels = color.label2rgb(labels, bg_label=10)
    if DEBUG:
        print(np.unique(labels))
    if separate:
        used_labels = np.unique(labels)[2:]  # skip bg
        nb_masks = len(used_labels)
        rows, cols = mask.shape[:2]
        masks = np.zeros((nb_masks, rows, cols), dtype=bool)
        for i, l in enumerate(used_labels):
            masks[i] = labels == l
        return masks
    else:
        return labels
os.makedirs('vis_watershed', exist_ok=True)
for p in tqdm(glob.glob("/Users/macbook/works/vesuvius/data/vasculature/masks/*")[2:50]):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    
    markers = find_masks_watershed(img, separate=False)
    # img[markers == -1] = [0, 255, 0]

    fig, axes = plt.subplots(ncols=2, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(markers, )
    ax[1].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()
    # plt.savefig(f"vis_watershed/{os.path.basename(p)}")
    break