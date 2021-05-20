from data.config import voc
import torch


def trim(params, iou_thresh=0.8):

    def iou(b1, b2):
        h1, w1 = b1
        h2, w2 = b2
        intersection = min(h1, h2) * min(w1, w2)
        return intersection / (h1 * w1 + h2 * w2 - intersection)

    res = []
    for p in params:
        p = p[p[:, -1].sort(descending=True)[1]]
        msk = torch.ones(p.size(0), dtype=torch.bool)
        for i in range(p.size(0)):
            if msk[i] == 0:
                continue
            # keep p[i], trim all the similar priors
            for j in range(i + 1, p.size(0)):
                if iou(p[i][:2], p[j][:2]) > iou_thresh:
                    msk[j] = 0
        res += [p[msk]]
    return res


# def gen_priors(params, num_types=32, cfg=voc):
#     params = trim(params, iou_thresh=0.75)
#     bbox = [p[:, :2] for k, p in enumerate(params)]
#     return bbox


def gen_priors(params, num_types=32, cfg=voc):
    if isinstance(params, str):
        params = torch.load(params)
    params = trim(params, iou_thresh=0.75)
    means = [p[:, -1].mean().item() for p in params]
    weights = torch.tensor(means).softmax(dim=0).tolist()
    # weights = [m / sum(means) for m in means]
    nums = [min(int(round(w * num_types)), params[k].size(0)) for k, w in enumerate(weights)]
    print('weights by layer: { %s }' % ', '.join(['%.3f' % w for w in weights]))
    print('types by layer: { %s }' % ', '.join(['%d' % n for n in nums]))
    print('%d types of priors in total' % sum(nums))
    print('%d priors in total' % (sum([nums[k] * n * n for k, n in enumerate(cfg['feature_maps'])])))
    # sort by alpha for every layer
    # params = [p[p[:, -1].sort(descending=True)[1]] for p in params]
    # keep top-k priors in each layer
    bbox = [p[:nums[k], :2] for k, p in enumerate(params)]
    return bbox