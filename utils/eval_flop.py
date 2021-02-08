# default dilation = 1, equal to the default settings of torch
def compute_next_size(input_size, kernel_size, stride=1, padding=0, dilation=1):
    kernel_size += (dilation - 1) * (kernel_size - 1)
    return (input_size - kernel_size + 2 * padding) // stride + 1


def eval_conv(in_size, in_chnl, out_chnl, kernel_size, stride=1, padding=0, dilation=1, relu=True, groups=1, bias=True):
    ret, out_size = __eval_conv(in_size, in_chnl / groups, out_chnl / groups,
                                kernel_size, stride, padding, dilation, bias)
    ret *= groups
    if relu:
        ret += eval_relu(out_size, out_chnl)[0]
    return ret, out_size


def eval_conv_bnrelu(in_size, in_chnl, out_chnl, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    ret, out_size = eval_conv(in_size, in_chnl, out_chnl, kernel_size, stride, padding, dilation, True, groups, False)
    return ret + eval_batch_norm(out_size, out_chnl)[0], out_size


def __eval_conv(in_size, in_chnl, out_chnl, kernel_size, stride=1, padding=0, dilation=1, bias=True):
    out_size = compute_next_size(in_size, kernel_size, stride, padding, dilation)
    if bias:
        cal_per_elem = 2 * kernel_size * kernel_size
    else:
        cal_per_elem = (2 * kernel_size * kernel_size - 1)
    ret = cal_per_elem * out_size * out_size * out_chnl * in_chnl
    return ret, out_size


def eval_relu(in_size, in_chnl):
    return in_size * in_size * in_chnl, in_size


def eval_fc(in_features, out_features):
    return 2 * in_features * out_features


def eval_pooling(in_size, in_chnl, kernel_size=2, stride=2, padding=0):
    size = compute_next_size(in_size, kernel_size, stride, padding)
    return size * size * in_chnl * kernel_size * kernel_size, size


def eval_batch_norm(in_size, in_channels, track_running_stats=True):
    # calculate mean
    ret = in_size * in_size
    # all elements minus mean
    ret += in_size * in_size
    # calculate standard variance
    ret += in_size * in_size * 2
    # calculate y: y = (x_ij - mean) / sqrt(var + e) * b + c = x'_ij * b' + c
    ret += in_size * in_size * 2
    # track running status y = y_old * (1 - a) + y * a
    if track_running_stats:
        ret += in_size * in_size * 3
    return ret * in_channels, in_size
