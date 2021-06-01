"""COCO Dataset Classes
from VOC Dataset Classes
Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Pan Heng
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

COCO18_CLASSES = (  # always index 0
    'horse', 'train', 'motorcycle', 'cat', 'bus',
    'cow', 'bird', 'chair', 'potted plant', 'bottle',
    'boat', 'car', 'dining table', 'sheep',
    'person', 'airplane', 'dog', 'bicycle')

COCO_CLASSES = (
    'train', 'truck', 'kite', 'microwave', 'book', 'scissors', 'sink', 'knife', 'suitcase', 'mouse', 'tennis racket',
    'cow', 'fork', 'potted plant', 'dining table', 'handbag', 'bird', 'refrigerator', 'traffic light', 'vase',
    'giraffe', 'umbrella', 'cup', 'bed', 'tv', 'sandwich', 'sheep', 'baseball glove', 'carrot', 'bowl', 'skis',
    'couch', 'laptop', 'cat', 'hot dog', 'fire hydrant', 'toilet', 'skateboard', 'frisbee', 'pizza', 'motorcycle',
    'backpack', 'bottle', 'toothbrush', 'sports ball', 'donut', 'apple', 'hair drier', 'bicycle', 'clock', 'toaster',
    'elephant', 'spoon', 'zebra', 'surfboard', 'bear', 'orange', 'person', 'car', 'tie', 'dog', 'parking meter',
    'cell phone', 'snowboard', 'bus', 'boat', 'baseball bat', 'horse', 'airplane', 'oven', 'bench', 'cake', 'stop sign',
    'banana', 'keyboard', 'wine glass', 'teddy bear', 'chair', 'broccoli', 'remote'
)

# note: if you used our download scripts, this should be right
COCO_ROOT = osp.join(HOME, "E:\\coco\\")


class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of COCO's 18 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        if class_to_ind == 'COCO18':
            class_to_ind = dict(zip([o.replace(' ', '') for o in COCO18_CLASSES], range(len(COCO18_CLASSES))))
        self.class_to_ind = class_to_ind or dict(
            zip([o.replace(' ', '') for o in COCO_CLASSES], range(len(COCO_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            label_idx = self.class_to_ind[name.replace(' ', '')]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class COCODetection(data.Dataset):
    """COCO Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to COCO folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 image_sets=(('18', 'trainval'),),
                 transform=None, target_transform=COCOAnnotationTransform(),
                 dataset_name='COCO'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for year, name in image_sets:
            rootpath = osp.join(self.root, 'coco' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.tensor(self.pull_image(index)).unsqueeze_(0)
