from data.config import voc
from layers.functions.prior_box import AdaptivePriorBox
import math
import torch
from torch import optim
import torch.nn.functional as F
from utils.analytical_utils import *
from utils.anchor_generator_utils import gen_priors
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

regression = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
gts_pth = r'truths\gts_voc07test.pth'
baseline_pth = r'anchors\voc_baseline.pth'
params_pth = r'selection\selected_priors_voc2.1.pth'
params_pth_lst = [
    'params\\packup\\params_test2_.pth',
    'params\\packup\\params_test3_k=20.pth',
    'params\\packup\\params_test4_k=40.pth',
    'params\\packup\\params_test5_k=30.pth',
    'params\\packup\\params_test5.1_k=30.pth',
    'params\\packup\\params_test6.1_k=25.pth',
    'params\\packup\\params_test7.1_k=10.pth',
    'params\\packup\\params_test8.1_k=20.pth',
    'params\\packup\\params_test10.1_k=20.pth',
    'selection\\selected_priors_voc2.1.pth'
]


def main():
    if regression:
        w = torch.ones(7, requires_grad=True)
        # w = torch.load('weights/pred_mAP_l1.pth')
        # w.requires_grad = True
        with torch.no_grad():
            w += 1. * (torch.rand(7) - 0.5)
        # gamma = weights[-1]
        # w = weights[:-1]
        # w = torch.tensor([0.21266897022724152, 0.3098817765712738, 1.6114717721939087, 5.077290058135986,
        #                   -0.6393837928771973, -1.5633304119110107, 0.792957067489624, 3.666994094848633])
        with open('data\\anchors_data.txt', 'r') as f:
            lines = f.readlines()
            table = torch.tensor([[float(o) for o in l.split()] for l in lines])
        A = table[:, :-1]
        y = table[:, -1]
        y = torch.exp(y)
        opt = optim.Adam([w], lr=0.001)
        for i in range(200000):
            loss = F.smooth_l1_loss(((w[:-1] * A).sum(dim=1) + w[-1]), y, reduction='sum')
            loss.backward()
            opt.step()
            opt.zero_grad()
            if (i + 1) % 100 == 0:
                print('iter %d: loss= %.8f' % (i + 1, loss.item()))
        print('weights:\n' + '\t'.join(['%.4f' % o for o in w.tolist()]))
        with torch.no_grad():
            pred_y = ((w[:-1] * A).sum(dim=1) + w[-1])
            # y = torch.exp(y)
            # pred_y = torch.exp(pred_y)
        print('Y\t\tY*')
        for i, ii in zip(y.tolist(), pred_y.tolist()):
            print('%.4f\t%.4f' % (i, ii))
        torch.save(w, 'weights/r_pred_mAP_l1_3.pth')
    else:
        results = []
        gts = torch.load(gts_pth).cuda()
        gen = AdaptivePriorBox(voc, phase='test')
        print('\t\tnum anchs\tloss\t\tpower1/3\tmean iou\trecall\t\tpower3')
        template = '\t%.0f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f'

        anchs = torch.load(baseline_pth).cuda().double()
        results.append(_analyze(anchs, gts, False))
        print('bl' + template % tuple(results[-1].tolist()))

        for pth in params_pth_lst[:-1]:
            params = gen_priors(torch.load(pth), 32, log=False)
            anchs = gen.forward(params).double()
            results.append(_analyze(anchs, gts, False))
            print('t' + pth[25:].split('_')[0] + template % tuple(results[-1].tolist()))

        pth = params_pth_lst[-1]
        params = torch.load(pth)
        anchs = gen.forward(params).double()
        results.append(_analyze(anchs, gts, False))
        print('voc2' + template % tuple(results[-1].tolist()))
        results = torch.stack(results)
        # torch.save(results, 'data\\anchs_analysis.pth')


def _analyze(anchs, gts, log=True):
    _t = AnchorsAnalyzer(anchs, gts)
    ret = torch.tensor([
        _t.get_num_anchors(),
        _t.get_approx_loss(),
        _t.get_power_mean(1 / 3),
        _t.get_mean_best_ious(),
        _t.get_recall(),
        _t.get_specialty()
    ])
    if log:
        print('num anchors = %.0f' % ret[0].item())
        print('approx loss = %.4f' % ret[1].item())
        print('mean log    = %.4f' % ret[2].item())
        print('mean best   = %.4f' % ret[3].item())
        print('recall      = %.4f' % ret[4].item())
        print('specialty   = %.4f' % ret[5].item())
    return ret


if __name__ == '__main__':
    main()
