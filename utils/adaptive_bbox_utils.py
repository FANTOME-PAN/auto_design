from data.config import voc
import torch


def gen_priors(params, num_types=32, cfg=voc):
    means = [p[:, -1].mean().item() for p in params]
    weights = [m / sum(means) for m in means]
    nums = [int(round(w * num_types)) for w in weights]
    print('weights by layer: { %s }' % ', '.join(['%.3f' % w for w in weights]))
    print('types by layer: { %s }' % ', '.join(['%d' % n for n in nums]))
    print('%d types of priors in total' % sum(nums))
    print('%d priors in total' % (sum([nums[k] * n * n for k, n in enumerate(cfg['feature_maps'])])))
    # sort by alpha for every layer
    params = [p[p[:,-1].sort(descending=True)[1]] for p in params]
    # keep top-k priors in each layer
    bbox = [p[:nums[k], :2] for k, p in enumerate(params)]
    return bbox