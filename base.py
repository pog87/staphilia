import numpy as np
import torch
import matplotlib.pylab as plt
import imea

def bb_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])


    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    #iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = interArea / float(boxAArea)

    # return the intersection over union value
    return iou

def area(box):
    x0,y0,x1,y1 = box
    return abs(x0-x1) * abs(y0-y1)

def sort_and_deduplicate(boxes, iou_threshold):
    # Sort the boxes list based on the area value
    sorted_boxes = sorted(boxes, key=lambda x: area(x), reverse=False)

    # Deduplicate masks based on the given iou_threshold
    filtered_boxes = []
    idx = []
    ix = 0
    for box in sorted_boxes:
        duplicate = False

        for fbox in filtered_boxes:
            if bb_iou(fbox, box) > iou_threshold:
                duplicate = True
                break

        if not duplicate:
            filtered_boxes.append(list(box))
            idx.append(ix)
        ix += 1
    print(len(sorted_boxes), len(filtered_boxes))

    return torch.Tensor(filtered_boxes), idx


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 10/255, 10/255, 0.4])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
    ax.text(x0, y0, label, color="grey")


def calc_shape_feats(mask):
    df_2d = imea.shape_measurements_2d(mask,1)

    elongation = ( df_2d.major_axis_length / df_2d.minor_axis_length).iloc[0]
    sphericity = (2 * np.sqrt(np.pi * np.sum(mask)) / df_2d.perimeter).iloc[0]
    convex_sphericity =  (2 * np.sqrt(np.pi * df_2d.area_convex) / df_2d.convex_perimeter).iloc[0]
    diameter_eq_area = (df_2d.diameter_equal_area / np.sqrt(np.pi * df_2d.area_projection)).iloc[0]
    diameter_eq_perim = (df_2d.diameter_equal_perimeter / df_2d.perimeter).iloc[0]

    nerosions_norm = (df_2d.n_erosions / np.sqrt(np.pi * np.sum(mask))).iloc[0]

    fractal1 = (df_2d.fractal_dimension_boxcounting_method).iloc[0]
    fractal2 = (df_2d.fractal_dimension_perimeter_method).iloc[0]

    feret_norm = ((df_2d.feret_median - df_2d.feret_min) / (df_2d.feret_max - df_2d.feret_min)).iloc[0]
    feret_vc   = ((df_2d.feret_mean) / (df_2d.feret_std)).iloc[0]
    return [elongation, sphericity, convex_sphericity, diameter_eq_area, diameter_eq_perim, nerosions_norm, fractal1, fractal2, feret_norm, feret_vc]


def calc_color_feat(img, mask, orig_img):
    #intensity norm
    img_norm = (img - orig_img.mean()) /  orig_img.std()
    colors = list("rgb")
    out = []
    for i in range(3):
        img_c = img[:,:,i]
        img_n = img_norm[:,:,i]
        out.append(img_c[mask].mean())
        out.append(img_c[mask].std())
        out.append(img_n[mask].mean())
        out.append(img_n[mask].std())
    return out
