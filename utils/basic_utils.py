from torch import nn
import sys
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def adjust_learning_rate(optimizer, gamma, step, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = init_lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    # init.xavier_uniform(param)
    nn.init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def get_file_name_from_path(path: str):
    p = path
    return p[max(p.rfind('\\'), p.rfind('/')) + 1: p.rfind('.') if p.rfind('.') > 0 else None]


def parse_rec(filename, with_size=False):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    if with_size:
        obj = tree.find('size')
        w = int(obj.find('width').text)
        h = int(obj.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        tmp = obj.find('truncated')
        obj_struct['truncated'] = int(tmp.text) if tmp is not None else None
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)
    if with_size:
        return objects, w, h
    return objects
