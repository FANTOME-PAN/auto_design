from data.coco import COCO_ROOT
import json
import torch
import numpy as np


## 修改这里的绝对地址 ##
## CHANGE PATH HERE ##
detection_results_path = r'eval\detection_results2021-08-21_11-06.pth'
save_path = r'.\detections_val2017_ssdopt_results.json'
##       END        ##
# anno_pth =COCO_ROOT + r'coco2017\Annotations\image_info_test-dev2017.json'
anno_pth = r'D:\COCO\annotations_train2017\annotations\instances_val2017.json'
with open(anno_pth, 'r') as f:
    anno = json.load(f)
img_name2id = dict([(o['file_name'].rstrip('.jpg'), o['id']) for o in anno['images']])
cat_name2id = dict([(o['name'], o['id']) for o in anno['categories']])

dets = torch.load(detection_results_path)
formatted = []
for det in dets:
    img_name = det['img_name']
    cat_name = det['cat_name']
    xmin, ymin, xmax, ymax = det['bbox']
    score = det['score']
    if isinstance(score, str):
        score = float(score)
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
    x = round(xmin, 1)
    y = round(ymin, 1)
    width = round(xmax - xmin, 1)
    height = round(ymax - ymin, 1)
    formatted.append({
        'image_id': img_name2id[img_name],
        'category_id': cat_name2id[cat_name],
        'bbox': [x, y, width, height],
        'score': round(score, 3)
    })
del dets
# keep top-k bboxes, k=100
k = 100
id2bboxes = dict()
for i, d in enumerate(formatted):
    if d['image_id'] not in id2bboxes.keys():
        id2bboxes[d['image_id']] = []
    id2bboxes[d['image_id']].append(i)
for img_id, ids in id2bboxes.items():
    id2bboxes[img_id] = np.array(ids)[np.array([formatted[i]['score'] for i in ids]).argsort()[::-1]][:k]

formatted = np.array(formatted)[np.concatenate(list(id2bboxes.values()))].tolist()

with open(save_path, 'w') as f:
    json.dump(formatted, f, separators=(',', ':'))
