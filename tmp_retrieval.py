import sys
import os
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {
            'name': obj.find('name').text,
            'pose': obj.find('pose').text
        }
        tmp = obj.find('truncated')
        obj_struct['truncated'] = int(tmp.text) if tmp is not None else None
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


if __name__ == '__main__':
    root = r'E:\coco\coco2014\Annotations'
    lst = os.listdir(root)
    class_set = set()
    cnt = 0
    for anno in lst:
        cnt += 1
        # print('read %s' % anno)
        objs = parse_rec('%s\\%s' % (root, anno))
        num = len(class_set)
        class_set.update([o['name'] for o in objs])
        if len(class_set) > num:
            print(class_set)
            print('number of classes: %d' % len(class_set))
        if cnt == 1000:
            print('-1000 xmls')
            cnt = 0

    print('all classes:')
    print('\n'.join(class_set))


