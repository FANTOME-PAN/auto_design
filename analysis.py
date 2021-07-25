from data.config import voc
from layers.functions.prior_box import AdaptivePriorBox
import torch
from utils.analytical_utils import *
from utils.anchor_generator_utils import gen_priors


torch.set_default_tensor_type('torch.cuda.FloatTensor')
gts_pth = r'E:\hwhit aiot project\auto_design\truths\gts_voc07test.pth'
anchs_pth = r'anchors\voc_baseline.pth'
params_pth = r'params\packup\params_test5_k=30.pth'

if __name__ == '__main__':
    gts = torch.load(gts_pth).cuda()
    # anchs = torch.load(anchs_pth).double().cuda()
    params = gen_priors(torch.load(params_pth), 32)
    gen = AdaptivePriorBox(voc, phase='test')
    anchs = gen.forward(params).double()
    aa = AnchorsAnalyzer(anchs, gts)
    tmp = aa.get_mean_best_ious()
    print('mean best = %.4f' % aa.get_mean_best_ious())
    print('recall    = %.4f' % aa.get_recall())
    print('specialty = %.4f' % aa.get_specialty())
