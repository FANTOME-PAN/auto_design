from data.coco import COCO_ROOT
import json
import torch


## 修改这里的绝对地址 ##
## CHANGE PATH HERE ##
detection_results_path = r''
save_path = r'.\detections_test-dev2017_ssdopt_results.json'
##       END        ##

with open(COCO_ROOT + r'coco2017\Annotations\image_info_test-dev2017.json', 'r') as f:
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
    x = (xmax + xmin) / 2.
    y = (ymax + ymin) / 2.
    width = xmax - xmin
    height = ymax - ymin
    formatted.append({
        'image_id': img_name2id[img_name],
        'category_id': cat_name2id[cat_name],
        'bbox': [round(x, 1), round(y, 1), round(width, 1), round(height, 1)],
        'score': score
    })

with open(save_path, 'w') as f:
    json.dump(formatted, f)
