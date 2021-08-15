from itertools import product
from math import sqrt
import torch
from torch.nn import Module

torch_bool = (torch.ones(1) > 0.).dtype


class IOAdapter:
    def __init__(self, cfg, phase='train', *args):
        self.cfg = cfg
        self.phase = phase
        pass

    def load(self, *args):
        pass

    def fit_input(self):
        pass

    def fit_output(self, msk=None):
        pass


class IOAdapterSSD(IOAdapter):
    def __init__(self, cfg, phase='train', random_range=0.2):
        super().__init__(cfg, phase)
        self.rd = random_range
        self.anchors = None
        self.anch2fmap = None
        self.fmap2locs = None
        self.msks = None
        self.__fit_in = False

    def load(self, anchs: torch.Tensor, anch2fmap, fmap2locs, msks):
        self.anchors, self.anch2fmap, self.fmap2locs, self.msks = anchs, anch2fmap, fmap2locs, msks
        self.anchors.requires_grad = self.phase == 'train'
        self.__fit_in = True

    def __build_template(self):
        fmap2locs = dict([((f, f), []) for f in self.cfg['feature_maps']])
        a2f = dict()
        anchors_template = []
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
            a2f[len(anchors_template)] = (f, f)
            anchors_template.append(torch.tensor([s_k, s_k]))
            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = sqrt(s_k * (maxs[k] / size))
            a2f[len(anchors_template)] = (f, f)
            anchors_template.append(torch.tensor([s_k_prime, s_k_prime]))

            # rest of aspect ratios
            for ar in ratios[k]:
                a2f[len(anchors_template)] = (f, f)
                anchors_template.append(torch.tensor([s_k * sqrt(ar), s_k / sqrt(ar)]))
                a2f[len(anchors_template)] = (f, f)
                anchors_template.append(torch.tensor([s_k / sqrt(ar), s_k * sqrt(ar)]))

            for i, j in product(range(f), repeat=2):
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                fmap2locs[(f, f)].append(torch.tensor([cx, cy]))

        for key in fmap2locs.keys():
            fmap2locs[key] = torch.stack(fmap2locs[key]).clamp(max=1., min=0.)
        anchors_template = torch.stack(anchors_template)
        return anchors_template, a2f, fmap2locs

    def fit_input(self, muls=(1, 2, 3, 4, 5, 6, 7, 8)):
        if self.__fit_in:
            return self.anchors, self.anch2fmap, self.fmap2locs, self.msks
        anch_template, a2f, fmap2locs = self.__build_template()
        # init params
        p = 0
        t_size = anch_template.size()[0]
        anchs = torch.zeros(t_size * sum(muls), 2, requires_grad=True)
        msks = [torch.zeros(t_size * sum(muls), dtype=torch_bool) for _ in range(len(muls))]
        anch2fmap = dict()
        with torch.no_grad():
            for i, m in enumerate(muls):
                tmp_anchs = anch_template.repeat(m, 1)
                sz = tmp_anchs.size()[0]
                # assign values to anchs
                anchs[p: p + sz] = tmp_anchs

                # mapping
                for j in range(sz):
                    anch2fmap[p + j] = a2f[j % t_size]
                # create mask
                msks[i][p: p + sz] = 1
                # mv pointer
                p += tmp_anchs.size()[0]
            # random
            rd = 1. + (2. * torch.rand(anchs.size()) - 1.) * self.rd
            anchs *= rd
        self.anchors, self.anch2fmap, self.fmap2locs, self.msks = anchs, anch2fmap, fmap2locs, msks
        self.__fit_in = True
        return anchs, anch2fmap, fmap2locs, msks

    def fit_output(self, msk=None) -> torch.Tensor:
        if not self.__fit_in:
            self.fit_input()
        if msk is None:
            msk = torch.ones(self.anchors.size()[0])
        msk = torch.nonzero(msk, as_tuple=False).flatten().tolist()
        # reverse dict
        fmap2anchs = dict(zip(self.fmap2locs.keys(), [[] for _ in range(len(self.fmap2locs))]))
        for i in msk:
            fmap2anchs[self.anch2fmap[i]].append(self.anchors[i])
        for fmap, lst in fmap2anchs.items():
            fmap2anchs[fmap] = torch.stack(lst)
        # build anchors
        ret = []
        for fmap, locs in self.fmap2locs.items():
            assert isinstance(locs, torch.Tensor)
            wh = locs.size()[0]
            n = fmap2anchs[fmap].size()[0]
            # wh * 2 => wh * 1 * 2 => wh * n * 2
            left2 = locs.unsqueeze(1).expand(wh, n, 2)
            # n * 2 => 1 * n * 2 => wh * n * 2
            right2 = fmap2anchs[fmap].unsqueeze(0).expand(wh, n, 2)
            ret.append(torch.cat([left2, right2], dim=2).view(-1, 4))
        return torch.cat(ret, dim=0)
