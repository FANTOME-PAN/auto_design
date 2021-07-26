from data.config import voc
from layers.functions.prior_box import AdaptivePriorBox
import math
import torch
from torch import optim
import torch.nn.functional as F
from utils.analytical_utils import *
from utils.anchor_generator_utils import gen_priors


regression = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')
gts_pth = r'E:\hwhit aiot project\auto_design\truths\gts_voc07test.pth'
anchs_pth = r'anchors\voc_baseline.pth'
params_pth = r'selection\selected_priors_voc2.1.pth'

if __name__ == '__main__':
    if regression:
        w = torch.ones(6, requires_grad=True)
        # w = torch.load('weights/pred_mAP_l1.pth')
        # w.requires_grad = True
        with torch.no_grad():
            w[1] = -1.
            w += 0.8 * (torch.rand(6) - 0.5)
        # w = torch.tensor([0.21266897022724152, 0.3098817765712738, 1.6114717721939087, 5.077290058135986,
        #                   -0.6393837928771973, -1.5633304119110107, 0.792957067489624, 3.666994094848633])
        with open('data\\anchors_data_reduced.txt', 'r') as f:
            lines = f.readlines()
            table = torch.tensor([[float(o) for o in l.split()] for l in lines])
        A = table[:, :-1]
        y = table[:, -1]
        # y = torch.exp(y) ** math.log(10.)
        opt = optim.Adam([w], lr=0.001)
        for i in range(100000):
            loss = F.smooth_l1_loss(((w[:-2] * A).sum(dim=1) + w[-2]) ** w[-1], y, reduction='sum')
            loss.backward()
            opt.step()
            opt.zero_grad()
            if (i + 1) % 100 == 0:
                print('iter %d: loss= %.8f' % (i + 1, loss.item()))
        print(w.tolist())
        with torch.no_grad():
            pred_y = ((w[:-2] * A).sum(dim=1) + w[-2]) ** w[-1]
            # y = torch.exp(y) ** math.log(10.)
            # pred_y = torch.exp(pred_y) ** math.log(10.)
            print(y)
            print(pred_y)
        torch.save(w, 'weights/r_pred_mAP_l1_3.pth')
    else:
        gts = torch.load(gts_pth).cuda()
        # anchs = torch.load(anchs_pth).double().cuda()
        # params = gen_priors(torch.load(params_pth), 32)
        params = torch.load(params_pth)
        # tpl = [p.size(0) for p in params]
        # print('types per layer: [%s]' % ', '.join([str(o) for o in tpl]))
        gen = AdaptivePriorBox(voc, phase='test')
        anchs = gen.forward(params).double()
        aa = AnchorsAnalyzer(anchs, gts)
        print('num anchors = %d' % aa.get_num_anchors())
        print('approx loss = %.4f' % aa.get_approx_loss())
        print('mean log    = %.4f' % aa.get_mean_log_ious())
        print('mean best   = %.4f' % aa.get_mean_best_ious())
        print('recall      = %.4f' % aa.get_recall())
        print('specialty   = %.4f' % aa.get_specialty())
