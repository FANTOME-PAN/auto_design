from utils.evaluations import parse_rec
import pickle


# too little images of one target
# there around 14k images from 80 classes, while only 8 classes have over 500 images and 2 classes have over 1k
# considering the given size of the dataset, I do not recommend pretraining the basenet on COCO.
def make_dataset_for_pretrain():
    coco_dir = 'E:/coco/coco2014/'
    with open(coco_dir + 'ImageSets/Main/trainval.txt', 'r') as f:
        id_lst = f.read().split()
    anno_pth = coco_dir + 'Annotations/%s.xml'
    img_pth = coco_dir + 'JPEGImages/%s.jpg'
    gt_lst = []
    for i, d in enumerate(id_lst):
        objs = parse_rec(anno_pth % d)
        if (i + 1) % 500 == 0:
            print('%d / %d' % (i + 1, len(id_lst)))
        if len(objs) != 1:
            continue
        gt_lst.append((d, objs[0]['name']))
    with open('one_target_annos.pkl', 'wb') as f:
        pickle.dump(gt_lst, f)
    with open('one_target_annos.txt', 'w') as f:
        f.write('\n'.join(['%s %s' % o for o in gt_lst]))


make_dataset_for_pretrain()

