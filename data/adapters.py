from itertools import product
from math import sqrt
import torch
from torch.nn import Module


class InputAdapter:
    def __init__(self, cfg=None):
        self.cfg = cfg
        pass

    def fit_input(self):
        pass

    def fit_output(self):
        pass


class InputAdapterSSD(InputAdapter):
    def fit_input(self):
        fmap2locs = dict([((f, f), []) for f in self.cfg['feature_maps']])
        anch2fmap = dict()
        anchs = []
        size = self.cfg['min_dim']
        mins = self.cfg['min_sizes']
        maxs = self.cfg['max_sizes']
        steps = self.cfg['steps']
        ratios = self.cfg['aspect_ratios']
        for k, f in enumerate(self.cfg['feature_maps']):
            f_k = size / steps[k]

            # aspect_ratio: 1
            # rel size: min_size
            s_k = mins[k] / size
            anch2fmap[len(anchs)] = (f, f)
            anchs.append(torch.tensor([s_k, s_k]))
            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = sqrt(s_k * (maxs[k] / size))
            anch2fmap[len(anchs)] = (f, f)
            anchs.append(torch.tensor([s_k_prime, s_k_prime]))

            # rest of aspect ratios
            for ar in ratios[k]:
                anch2fmap[len(anchs)] = (f, f)
                anchs.append(torch.tensor([s_k * sqrt(ar), s_k / sqrt(ar)]))
                anch2fmap[len(anchs)] = (f, f)
                anchs.append(torch.tensor([s_k / sqrt(ar), s_k * sqrt(ar)]))

            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                fmap2locs[(f, f)].append(torch.tensor([cx, cy]))

        for key in fmap2locs.keys():
            fmap2locs[key] = torch.stack(fmap2locs[key])
        anchs = torch.stack(anchs)
        return anchs, anch2fmap, fmap2locs



