from PIL import Image
import requests
import torch
import numpy as np
import matplotlib.pylab as plt
import imea

from transformers import AutoProcessor, GroundingDinoForObjectDetection
from transformers import SamModel, SamProcessor

from base import show_box, show_mask, sort_and_deduplicate
from base import calc_shape_feats, calc_color_feat

import pickle, sys, os

fname = sys.argv[1]

filename, file_extension = os.path.splitext(fname)

image = Image.open(fname)
text = "wine grapes."


## DINO

processor_dino = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model_dino = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

inputs = processor_dino(images=image, text=text, return_tensors="pt")
outputs = model_dino(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = processor_dino.image_processor.post_process_object_detection(
    outputs,
    threshold=0.25,
    target_sizes=target_sizes)[0]

scores, labels, boxes = results["scores"], results["labels"], results["boxes"]



for score, label, box in zip(scores, labels, boxes):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {label.item()} with confidence " f"{round(score.item(), 3)} at location {box}")


boxes_filt, idx = sort_and_deduplicate(boxes,0.6)

##scores = scores[[idx]]

## SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
model_sam = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor_sam = SamProcessor.from_pretrained("facebook/sam-vit-huge")


raw_image = image.convert("RGB")

inputs = processor_sam(raw_image, input_boxes=[boxes_filt.tolist()], return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model_sam(**inputs)

masks = processor_sam.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)[0]
scores_sam = outputs.iou_scores


## Feature calculations and classification
classifier = pickle.load(open("classifier.pkl", 'rb'))

categs = ["FS", "VR","RP"]
prediction = []
for i in range(boxes_filt.shape[0]):

    box = boxes_filt[i]
    mask = masks[i]
    print(box, torch.sum(mask))
    x0,y0,x1,y1 = np.int_(box)

    x0 = max( 0, x0-10)
    y0 = max( 0, y0-10)
    x1 = min( mask.shape[-1], x1+10)
    y1 = min( mask.shape[-2], y1+10)

    Iz = np.array(image)[y0:y1, x0:x1, :]
    Mz = np.array(mask[0,y0:y1, x0:x1])
    Bz = [x0,y0,x1,y1]
    print(Mz.sum())

    X = np.array(calc_shape_feats(Mz)+calc_color_feat(Iz, Mz, np.array(image) ))


    cat = classifier.predict(X.reshape(1,-1))[0]
    prob = classifier.predict_proba(X.reshape(1,-1))[0][cat]

    prediction.append([cat,prob])




plt.figure()
plt.imshow(image)
for i in range(len(boxes_filt)):
    cat, prob = prediction[i]
    label = f"{categs[cat]} p={prob:0.3}"
    show_box(boxes_filt[i].detach().numpy(), plt.gca(), label)
    show_mask(masks[i,0,:], plt.gca(), random_color=True)
plt.axis('off')
#plt.show()
plt.savefig(filename+"_class"+".png", bbox_inches="tight")
#plt.close()
