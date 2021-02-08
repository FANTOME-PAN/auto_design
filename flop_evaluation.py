from utils.eval_flop import *


def eval_big_net():
    params = [0, 28]

    def block(in_chnl, out_chnl):
        cal, size = eval_conv(params[1], in_chnl, out_chnl, 3)
        size = compute_next_size(size, 2, 2)
        params[0] += cal
        params[1] = size

    block(1, 32)
    block(32, 64)
    block(64, 128)

    params[0] += eval_fc(128, 625)
    params[0] += eval_fc(625, 10)
    print('big net FLOP: %d' % params[0])
    return params[0]


def vgg16_conv(size=300):
    print('---- VGG 16 evaluation ----')
    total_flop = 0
    # block 1
    flop, size = eval_conv(size, 3, 64, kernel_size=3, stride=1, padding=1)
    tmp, size = eval_conv(size, 64, 64, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 64, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk1: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 2
    flop = 0
    tmp, size = eval_conv(size, 64, 128, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 128, 128, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 128, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk2: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 3
    flop = 0
    tmp, size = eval_conv(size, 128, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 256, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 256, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 256, kernel_size=2, stride=2, padding=1)
    flop += tmp
    print('blk3: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 4
    flop = 0
    tmp, size = eval_conv(size, 256, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 512, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk4: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 5
    flop = 0
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    print('blk5: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    print('vgg16 conv layers: total flop= %s' % format(total_flop, ','))
    return total_flop


def vgg11_conv(size=300):
    print('---- VGG 11 evaluation ----')
    total_flop = 0
    # block 1
    flop = 0
    tmp, size = eval_conv(size, 3, 64, kernel_size=3, stride=1, padding=1)
    # tmp, size = eval_conv(size, 64, 64, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 64, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk1: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 2
    flop = 0
    tmp, size = eval_conv(size, 64, 128, kernel_size=3, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_conv(size, 128, 128, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    tmp, size = eval_pooling(size, 128, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk2: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 3
    flop = 0
    tmp, size = eval_conv(size, 128, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 256, 256, kernel_size=3, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_conv(size, 256, 256, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    tmp, size = eval_pooling(size, 256, kernel_size=2, stride=2, padding=1)
    flop += tmp
    print('blk3: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 4
    flop = 0
    tmp, size = eval_conv(size, 256, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    tmp, size = eval_pooling(size, 512, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk4: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 5
    flop = 0
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    print('blk5: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    print('vgg11 conv layers: total flop= %s' % format(total_flop, ','))
    return total_flop


def vgg11_reduced(size=300):
    print('---- VGG 11 reduced evaluation ----')
    total_flop = 0
    # block 1
    flop = 0
    tmp, size = eval_conv(size, 3, 64, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 64, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk1: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 2
    flop = 0
    tmp, size = eval_conv(size, 64, 128, kernel_size=3, stride=2, padding=1)
    flop += tmp
    # tmp, size = eval_pooling(size, 128, kernel_size=2, stride=2, padding=0)
    # flop += tmp
    print('blk2: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 3
    flop = 0
    tmp, size = eval_conv(size, 128, 256, kernel_size=3, stride=2, padding=1)
    flop += tmp
    # tmp, size = eval_pooling(size, 256, kernel_size=2, stride=2, padding=1)
    # flop += tmp
    print('blk3: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 4
    flop = 0
    tmp, size = eval_conv(size, 256, 512, kernel_size=3, stride=2, padding=1)
    flop += tmp
    # tmp, size = eval_pooling(size, 512, kernel_size=2, stride=2, padding=0)
    # flop += tmp
    print('blk4: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 5
    flop = 0
    # tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    # print('blk5: flop = %15s || size = %3d' % (format(flop, ','), size))
    # total_flop += flop
    print('vgg11 conv layers: total flop= %s' % format(total_flop, ','))
    return total_flop


def conv6_7():
    print('---- conv6 and conv7 evaluation ----')
    flop = 0
    tmp, size = eval_pooling(19, 512, 3, 1, 1)
    flop += tmp
    tmp, size = eval_conv(19, 512, 1024, 3, padding=6, dilation=6)
    flop += tmp
    print('conv6 flop = %s' % format(tmp, ','))
    tmp, size = eval_conv(size, 1024, 1024, 1, 1, 0)
    flop += tmp
    print('conv7 flop = %s' % format(tmp, ','))
    print('total flop = %s' % format(flop, ','))
    return flop


def conv6_7_reduced():
    print('---- conv6 and conv7 reduced evaluation ----')
    flop = 0
    tmp, size = eval_pooling(19, 512, 3, 1, 1)
    flop += tmp
    tmp, size = eval_conv(19, 512, 512, 3, padding=6, dilation=6)
    flop += tmp
    print('conv6 flop = %s' % format(tmp, ','))
    tmp, size = eval_conv(size, 512, 1024, 1, 1, 0)
    flop += tmp
    print('conv7 flop = %s' % format(tmp, ','))
    print('total flop = %s' % format(flop, ','))
    return flop


def extra_feature_layers():
    print('---- extra feature layers evaluation ----')
    total_flop = 0

    name = 'Conv8_2'
    flop = 0
    _, size = eval_conv(19, 1024, 256, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 256, 512, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv9_2'
    flop = 0
    _, size = eval_conv(size, 512, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv10_2'
    flop = 0
    _, size = eval_conv(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv11_2'
    flop = 0
    _, size = eval_conv(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))
    print('total flop = %s' % format(total_flop, ','))
    return total_flop


def extra_feature_layers_reduced():
    print('---- extra feature layers reduced evaluation ----')
    total_flop = 0

    name = 'Conv8_2'
    flop = 0
    _, size = eval_conv(19, 1024, 256, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 256, 512, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv9_2'
    flop = 0
    _, size = eval_conv(size, 512, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv10_2'
    flop = 0
    _, size = eval_conv(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv11_2'
    flop = 0
    _, size = eval_conv(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))
    print('total flop = %s' % format(total_flop, ','))
    return total_flop


def classifiers(classes=5):
    print('---- classifiers evaluation ----')
    total = 0
    lst = [(38, 512, 4), (19, 1024, 6), (10, 512, 6), (5, 256, 6),
           (3, 256, 4), (1, 256, 4)]
    for i, (size, channels, prior_num) in enumerate(lst):
        flop = eval_conv(size, channels, prior_num * classes, 3, 1, 1)[0]
        flop += eval_conv(size, channels, prior_num * 4, 3, 1, 1)[0]
        total += flop
        print('Classifier %d: flop = %12s' % (i + 1, format(flop, ',')))
    print('total = %s' % format(total, ','))
    return total


def classifiers_reduced(classes=5):
    print('---- classifiers reduced evaluation ----')
    total = 0
    lst = [(19, 1024, 6), (10, 512, 6), (5, 256, 6),
           (3, 256, 4), (1, 256, 4)]
    for i, (size, channels, prior_num) in enumerate(lst):
        flop = eval_conv(size, channels, prior_num * classes, 3, 1, 1)[0]
        flop += eval_conv(size, channels, prior_num * 4, 3, 1, 1)[0]
        total += flop
        print('Classifier %d: flop = %12s' % (i + 1, format(flop, ',')))
    print('total = %s' % format(total, ','))
    return total


def conf_net():
    print('---- Confidence Net Evaluation ----')
    total = 0
    p = 5  # the bbox output may not be useful
    total += eval_fc(5 * 5 * 6 * p, 256)
    total += eval_fc(3 * 3 * 4 * p, 256)
    total += eval_fc(1 * 1 * 4 * p, 256)

    total += eval_fc(3 * 256, 2)
    print('flop = %s' % format(total, ','))


def eval_mobi_smallnet():
    def v2_blk(in_size, in_channels, out_channels, stride=1, expansion=6, padding=1):
        e_in = in_channels * expansion
        # expansion
        _flop, _out_size = eval_conv(in_size, in_channels, e_in, kernel_size=1)
        _flop += eval_batch_norm(_out_size, in_channels)[0]
        # depthwise conv
        t = eval_conv(_out_size, e_in, e_in, 3, stride=1, padding=1, groups=e_in) if stride == 1 \
            else eval_conv(_out_size, e_in, e_in, 3, stride=2, padding=padding, groups=e_in)
        _flop, _out_size = _flop + t[0], t[1]
        _flop += eval_batch_norm(_out_size, e_in)[0]
        # decrease dimensionality
        t = eval_conv(_out_size, e_in, out_channels, kernel_size=1, relu=False)
        _flop, _out_size = _flop + t[0], t[1]
        _flop += eval_batch_norm(_out_size, out_channels)[0]
        # skip connection
        if stride == 1 and in_channels == out_channels:
            _flop += _out_size * _out_size * out_channels
        return _flop, _out_size

    print('---- MobileNetV2 reduced evaluation ----')
    total_flop = 0
    size = 300
    # block 0
    flop = 0
    tmp, size = eval_conv(size, 3, 64, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = eval_pooling(size, 64, kernel_size=2, stride=2, padding=0)
    flop += tmp
    print('blk0: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 1
    flop = 0
    tmp, size = v2_blk(size, 64, 32, stride=1, padding=1, expansion=1)
    flop += tmp
    print('blk1: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 2
    flop = 0
    tmp, size = v2_blk(size, 32, 48, stride=2, padding=1)
    flop += tmp
    tmp, size = v2_blk(size, 48, 48, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_pooling(size, 128, kernel_size=2, stride=2, padding=0)
    # flop += tmp
    print('blk2: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 3
    flop = 0
    tmp, size = v2_blk(size, 48, 64, stride=2, padding=1)
    flop += tmp
    tmp, size = v2_blk(size, 64, 64, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_pooling(size, 256, kernel_size=2, stride=2, padding=1)
    # flop += tmp
    print('blk3: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 4
    flop = 0
    tmp, size = v2_blk(size, 64, 128, stride=2, padding=1)
    flop += tmp
    tmp, size = v2_blk(size, 128, 128, stride=1, padding=1)
    flop += tmp
    tmp, size = v2_blk(size, 128, 128, stride=1, padding=1)
    flop += tmp
    # tmp, size = eval_pooling(size, 512, kernel_size=2, stride=2, padding=0)
    # flop += tmp
    print('blk4: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 5
    flop = 0
    # tmp, size = eval_conv(size, 512, 512, kernel_size=3, stride=1, padding=1)
    # flop += tmp
    # print('blk5: flop = %15s || size = %3d' % (format(flop, ','), size))
    # total_flop += flop
    print('MobileNetV2: total flop= %s' % format(total_flop, ','))
    assert size == 19
    print('---- conv6 and conv7 with mobile net v2 ----')
    flop = 0
    tmp, size = eval_pooling(19, 128, 3, 1, 1)
    flop += tmp
    tmp, size = v2_blk(19, 128, 1024)
    flop += tmp
    print('conv6 flop = %s' % format(tmp, ','))
    tmp, size = eval_conv(size, 1024, 1024, 1, 1, 0)
    flop += tmp
    print('conv7 flop = %s' % format(tmp, ','))
    print('total flop = %s' % format(flop, ','))
    return total_flop + flop


def eval_mobi_smallnet_v2():
    def irb(in_size, in_channels, out_channels, stride=1, expansion=6, padding=1):
        e_in = in_channels * expansion
        # expansion
        _flop, _out_size = eval_conv(in_size, in_channels, e_in, kernel_size=1)
        _flop += eval_batch_norm(_out_size, in_channels)[0]
        # depthwise conv
        t = eval_conv(_out_size, e_in, e_in, 3, stride=1, padding=1, groups=e_in) if stride == 1 \
            else eval_conv(_out_size, e_in, e_in, 3, stride=2, padding=padding, groups=e_in)
        _flop, _out_size = _flop + t[0], t[1]
        _flop += eval_batch_norm(_out_size, e_in)[0]
        # decrease dimensionality
        t = eval_conv(_out_size, e_in, out_channels, kernel_size=1, relu=False)
        _flop, _out_size = _flop + t[0], t[1]
        _flop += eval_batch_norm(_out_size, out_channels)[0]
        # skip connection
        if stride == 1 and in_channels == out_channels:
            _flop += _out_size * _out_size * out_channels
        return _flop, _out_size

    def fdsb(in_size, in_channels, medium_channels, out_channels, padding=1):
        ret = [0, 0]
        q = eval_conv_bnrelu(in_size, in_channels, medium_channels, kernel_size=1)
        ret[0] = q[0]
        q = eval_conv_bnrelu(q[1], medium_channels, medium_channels, kernel_size=3,
                             stride=2, padding=padding, groups=medium_channels)
        ret[0] += q[0]
        q = eval_conv_bnrelu(q[1], medium_channels, out_channels, kernel_size=1)
        ret[0] += q[0]
        ret[1] = q[1]
        return ret[0], ret[1]

    def pred_blk(in_size, in_channels, out_channels):
        ret = eval_conv_bnrelu(in_size, in_channels, in_channels, kernel_size=3, stride=1,
                               padding=1, groups=in_channels)[0]
        ret += eval_conv(in_size, in_channels, out_channels, kernel_size=1, relu=False, bias=False)[0]
        ret += eval_batch_norm(in_size, out_channels)[0]
        return ret, in_size

    print('---- MobileNetV2 evaluation ----')
    total_flop = 0
    num_classes = 21
    size = 300
    # stage 1
    flop = 0
    tmp, size = eval_conv_bnrelu(size, 3, 32, kernel_size=3, stride=2, padding=1)
    flop += tmp
    tmp, size = irb(size, 32, 16, expansion=1)
    flop += tmp
    print('stage 1: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # stage 2
    flop = 0
    tmp, size = irb(size, 16, 24, stride=2)
    flop += tmp
    tmp, size = irb(size, 24, 24)
    flop += tmp
    # tmp, size = eval_pooling(size, 128, kernel_size=2, stride=2, padding=0)
    # flop += tmp
    print('stage 2: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # stage 3
    flop = 0
    tmp, size = irb(size, 24, 32, stride=2)
    flop += tmp
    tmp, size = irb(size, 32, 32)
    flop += tmp
    tmp, size = irb(size, 32, 32)
    flop += tmp
    # tmp, size = eval_pooling(size, 256, kernel_size=2, stride=2, padding=1)
    # flop += tmp
    print('stage 3: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # stage 4
    flop = 0
    tmp, size = irb(size, 32, 64, stride=2)
    flop += tmp
    tmp, size = irb(size, 64, 64)
    flop += tmp
    tmp, size = irb(size, 64, 64)
    flop += tmp
    tmp, size = irb(size, 64, 64)
    flop += tmp
    tmp, size = irb(size, 64, 96)
    flop += tmp
    tmp, size = irb(size, 96, 96)
    flop += tmp
    tmp, size = irb(size, 96, 96)
    flop += tmp
    print('stage 4: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # stage 5
    flop = 0
    tmp, size = irb(size, 96, 160, stride=2)
    flop += tmp
    tmp, size = irb(size, 160, 160)
    flop += tmp
    tmp, size = irb(size, 160, 160)
    flop += tmp
    assert size == 10
    tmp, size = irb(size, 160, 320)
    flop += tmp
    tmp, size = eval_conv_bnrelu(10, 320, 1280, kernel_size=1)
    flop += tmp
    print('stage 5: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop

    # extras
    flop = 0
    tmp, size = fdsb(size, 1280, 256, 512)
    flop += tmp
    tmp, size = fdsb(size, 512, 128, 256)
    flop += tmp
    tmp, size = fdsb(size, 256, 128, 256, padding=0)
    flop += tmp
    tmp, size = fdsb(size, 256, 64, 128)
    flop += tmp
    print('extras: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # loc layers
    flop = 0
    tmp, size = pred_blk(19, 576, 3 * 4)
    flop += tmp
    tmp, size = pred_blk(10, 1280, 6 * 4)
    flop += tmp
    tmp, size = pred_blk(5, 512, 6 * 4)
    flop += tmp
    tmp, size = pred_blk(3, 256, 6 * 4)
    flop += tmp
    tmp, size = pred_blk(1, 256, 6 * 4)
    flop += tmp
    tmp, size = pred_blk(1, 128, 6 * 4)
    flop += tmp
    print('loc layers: flop = %15s' % format(flop, ','))
    total_flop += flop
    # conf layers
    flop = 0
    tmp, size = pred_blk(19, 576, 3 * num_classes)
    flop += tmp
    tmp, size = pred_blk(10, 1280, 6 * num_classes)
    flop += tmp
    tmp, size = pred_blk(5, 512, 6 * num_classes)
    flop += tmp
    tmp, size = pred_blk(3, 256, 6 * num_classes)
    flop += tmp
    tmp, size = pred_blk(1, 256, 6 * num_classes)
    flop += tmp
    tmp, size = pred_blk(1, 128, 6 * num_classes)
    flop += tmp
    print('conf layers: flop = %15s' % format(flop, ','))
    total_flop += flop
    print('MobileNetV2: total flop= %s' % format(total_flop, ','))
    return total_flop


def mobilenet_ssd(size=300):
    def mconv(in_size, in_chnl, out_chnl, stride=1):
        __f, __s = eval_conv_bnrelu(in_size, in_chnl, in_chnl, 3, stride, 1, groups=in_chnl)
        __f += eval_conv_bnrelu(in_size, in_chnl, out_chnl, 1, stride)[0]
        return __f, __s

    print('---- MobileNet SSD evaluation(base) ----')
    total_flop = 0
    # block 1
    flop = 0
    tmp, size = eval_conv_bnrelu(size, 3, 32, kernel_size=3, stride=1, padding=1)
    flop += tmp
    tmp, size = mconv(size, 32, 64, stride=1)
    flop += tmp
    print('blk1: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 2
    flop = 0
    tmp, size = mconv(size, 64, 128, stride=2)
    flop += tmp
    tmp, size = mconv(size, 128, 128, stride=1)
    flop += tmp
    print('blk2: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 3
    flop = 0
    tmp, size = mconv(size, 128, 256, stride=2)
    flop += tmp
    tmp, size = mconv(size, 256, 256, stride=1)
    flop += tmp
    print('blk3: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 4
    flop = 0
    tmp, size = mconv(size, 256, 512, stride=2)
    flop += tmp
    tmp, size = mconv(size, 512, 512, stride=1)
    flop += tmp
    tmp, size = mconv(size, 512, 512, stride=1)
    flop += tmp
    tmp, size = mconv(size, 512, 512, stride=1)
    flop += tmp
    tmp, size = mconv(size, 512, 512, stride=1)
    flop += tmp
    tmp, size = mconv(size, 512, 512, stride=1)
    flop += tmp
    print('blk4: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    # block 5
    flop = 0
    tmp, size = mconv(size, 512, 1024, stride=2)
    flop += tmp
    tmp, size = mconv(size, 1024, 1024, stride=1)
    flop += tmp
    print('blk5: flop = %15s || size = %3d' % (format(flop, ','), size))
    total_flop += flop
    print('mobilenet ssd base: total flop= %s' % format(total_flop, ','))

    print('---- mobilenet ssd extra feature layers evaluation ----')
    name = 'Conv8_2'
    flop = 0
    _, size = eval_conv_bnrelu(19, 1024, 256, 1, 1, 0)
    flop += _
    _, size = eval_conv_bnrelu(size, 256, 512, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv9_2'
    flop = 0
    _, size = eval_conv_bnrelu(size, 512, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv_bnrelu(size, 128, 256, 3, 2, 1)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv10_2'
    flop = 0
    _, size = eval_conv_bnrelu(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv_bnrelu(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))

    name = 'Conv11_2'
    flop = 0
    _, size = eval_conv_bnrelu(size, 256, 128, 1, 1, 0)
    flop += _
    _, size = eval_conv_bnrelu(size, 128, 256, 3, 1, 0)
    flop += _
    total_flop += flop
    print('%-8s: flop = %12s || size = %2d' % (name, format(flop, ','), size))
    print('total flop = %s' % format(total_flop, ','))

    print('---- mobilenet ssd classifiers evaluation ----')
    total = 0
    lst = [(38, 512, 4), (19, 1024, 6), (10, 512, 6), (5, 256, 6),
           (3, 256, 4), (1, 256, 4)]
    for i, (size, channels, prior_num) in enumerate(lst):
        flop = eval_conv(size, channels, prior_num * 21, 1, relu=False)[0]
        flop += eval_conv(size, channels, prior_num * 4, 1, relu=False)[0]
        total += flop
        print('Classifier %d: flop = %12s' % (i + 1, format(flop, ',')))
    print('total = %s' % format(total, ','))
    total_flop += total

    return total_flop


big = vgg16_conv()
big += conv6_7()
big += extra_feature_layers()
big += classifiers(5)
# vgg11_conv()
# small = mobilenet_ssd()
small = eval_mobi_smallnet_v2()
# vgg11_reduced()
# conv6_7_reduced()
# small += extra_feature_layers_reduced()
# small += classifiers_reduced(5)
print('\n\nBig net flop = %s\nSmall net flop = %s\n ratio = %f' % (format(big, ','), format(small, ','),
                                                                   small / big))
conf_net()
