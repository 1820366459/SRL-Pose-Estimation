import torch
from torch import nn
from torch.nn import functional as F
import math


def get_nonlocal_block(block_type):
    block_dict = {'fft': FFTBlock, 'cgnl': SpatialCGNLx, 'nl': NonLocal,
                  'a2': DoubleAttention, 'cc': CCAttention, 'lo': LOBlock}
    if block_type in block_dict:
        return block_dict[block_type]
    else:
        raise ValueError("UNKOWN NONLOCAL BLOCK TYPE:", block_type)


def _init_conv(conv_layer, std=0.01, identity_init=True, kaiming_init=False, groups_identity_init=False):
    if groups_identity_init:
        s = conv_layer.weight.data.size()
        w = torch.eye(s[1])
        conv_layer.weight.data = torch.cat(
            [w] * (s[0] // s[1]), 0).view(conv_layer.weight.data.size())
    elif identity_init:
        w = torch.eye(conv_layer.weight.data.size()[0])
        conv_layer.weight.data = w.view(conv_layer.weight.data.size())
    elif kaiming_init:
        nn.init.kaiming_normal_(conv_layer.weight)
    else:
        nn.init.normal_(conv_layer.weight, 0.0, std)
    if len(list(conv_layer.parameters())) > 1:
        nn.init.constant_(conv_layer.bias, 0.0)


def _init_bn(bn_layer, zero_init=False):
    if zero_init:
        bn_layer.weight.data.zero_()
        bn_layer.bias.data.zero_()
    else:
        bn_layer.weight.data.fill_(1)
        bn_layer.bias.data.zero_()


def _init_nl(module):
    for name, m in module.named_modules():
        if len(m._modules) > 0:
            continue
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if len(list(m.parameters())) > 1:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 0)
            nn.init.constant_(m.bias, 0)
        elif len(list(m.parameters())) > 0:
            raise ValueError("UNKOWN NONLOCAL LAYER TYPE:", name, m)


def orth(origin):
    return origin.qr()[0]


class NonLocalModule(nn.Module):

    def __init__(self, in_channels, **kwargs):
        super(NonLocalModule, self).__init__()

    def init_modules(self):
        _init_nl(self)


class SpectrumReLU(nn.Module):

    def __init__(self, spatial_size):
        super(SpectrumReLU, self).__init__()

        self.weight = nn.Parameter(torch.ones(spatial_size))
        self.bias = nn.Parameter(torch.zeros(spatial_size))

    def forward(self, x):
        x = torch.add(torch.mul(self.weight, x), self.bias)
        x = nn.functional.relu(x, inplace=True)
        return x


class FFTBlock(NonLocalModule):

    def __init__(self, in_channels, group=8, spectrum_relu=False, spatial_size=None, **kwargs):
        # bn_layer not used
        super(FFTBlock, self).__init__(in_channels)
        self.groups = group
        self.alpha = nn.Parameter(torch.zeros(1))
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn = torch.nn.BatchNorm2d(in_channels * 2)
        if spectrum_relu:
            print('[*] Using spectrum_relu')
            spatial_size = (spatial_size[0], spatial_size[1] // 2 + 1)
            self.relu = SpectrumReLU(spatial_size)
        else:
            self.relu = torch.nn.ReLU(inplace=True)

    def init_modules(self):
        _init_conv(self.conv_layer, identity_init=False,
                   groups_identity_init=True)
        _init_bn(self.bn)

    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch, c, h, w = x.size()
        r_size = x.size()

        ffted = torch.rfft(x, signal_ndim=2)  # (batch, c, h, w/2+1, 2)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(ffted.size()[0:1] + (-1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view(ffted.size()[0:1] + (c, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        output = torch.irfft(ffted, signal_ndim=2, signal_sizes=r_size[2:])

        return self.alpha * output + x


class LOBlock(NonLocalModule):

    def __init__(self, in_channels, spatial_size, group=2, spectrum_relu=False, **kwargs):
        # bn_layer not used
        super(LOBlock, self).__init__(in_channels)
        self.groups = group
        self.source_P = nn.Parameter(
            orth(torch.randn((spatial_size[0], spatial_size[0]))))
        self.source_Q = nn.Parameter(
            orth(torch.randn((spatial_size[1], spatial_size[1]))))
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups)
        self.bn = nn.BatchNorm2d(in_channels, momentum=0.1)
        if spectrum_relu:
            print('[*] Using spectrum_relu')
            self.relu = SpectrumReLU(spatial_size)
        else:
            self.relu = torch.nn.ReLU(inplace=True)

    def init_modules(self):
        _init_conv(self.conv_layer, identity_init=False,
                   groups_identity_init=True)
        _init_bn(self.bn)

    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''
        batch, c, h, w = x.size()
        x_size = x.size()

        with torch.no_grad():
            self.source_P.data = orth(self.source_P.data)
            self.source_Q.data = orth(self.source_Q.data)

        P_r = self.source_P.transpose(0, 1)
        Q_r = self.source_Q.transpose(0, 1)

        trans = self.source_P.matmul(
            x.view((-1,) + x_size[2:])).matmul(self.source_Q)
        trans = trans.view(x_size)
        trans = self.conv_layer(trans)  # same as input
        trans = self.relu(self.bn(trans))

        trans = P_r.matmul(trans).matmul(Q_r)
        output = trans

        return x + output


class SpatialCGNL(NonLocalModule):
    """Spatial CGNL block with dot production kernel for image classfication.
    """

    def __init__(self, inplanes, use_scale=False, groups=8, **kwargs):
        planes = inplanes // 2
        self.use_scale = use_scale
        self.groups = groups

        super(SpatialCGNL, self).__init__(inplanes)
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                           groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

    def kernel(self, t, p, g, b, c, h, w):
        """The linear kernel (dot production).

        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        if self.use_scale:
            att = att.div((c * h * w)**0.5)

        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class SpatialCGNLx(NonLocalModule):
    """Spatial CGNL block with Gaussian RBF kernel for image classification.
    """

    def __init__(self, inplanes, use_scale=False, groups=8, order=3, **kwargs):
        planes = inplanes // 2
        self.use_scale = use_scale
        self.groups = groups
        self.order = order

        super(SpatialCGNLx, self).__init__(inplanes)
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                           groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

    def kernel(self, t, p, g, b, c, h, w):
        """The non-linear kernel (Gaussian RBF).

        Args:
            t: output of conv theata
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """

        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        # gamma
        gamma = t.new_tensor(torch.Tensor(1).fill_(1e-4))

        # NOTE:
        # We want to keep the high-order feature spaces in Taylor expansion to
        # rich the feature representation, so the l2 norm is not used here.
        #
        # Under the above precondition, the β should be calculated
        # by β = exp(−γ(∥θ∥^2 +∥φ∥^2)).
        # But in the experiments, we found training becomes very difficult.
        # So we simplify the implementation to
        # ease the gradient computation through calculating the β = exp(−2γ).

        # beta
        beta = torch.exp(-2 * gamma)

        t_taylor = []
        p_taylor = []
        for order in range(self.order + 1):
            # alpha
            alpha = torch.mul(
                torch.div(
                    torch.pow(
                        (2 * gamma),
                        order),
                    math.factorial(order)),
                beta)

            alpha = torch.sqrt(alpha)

            _t = t.pow(order).mul(alpha)
            _p = p.pow(order).mul(alpha)

            t_taylor.append(_t)
            p_taylor.append(_p)

        t_taylor = torch.cat(t_taylor, dim=1)
        p_taylor = torch.cat(p_taylor, dim=1)

        att = torch.bmm(p_taylor, g)

        if self.use_scale:
            att = att.div((c * h * w)**0.5)

        att = att.view(b, 1, int(self.order + 1))
        x = torch.bmm(att, t_taylor)
        x = x.view(b, c, h, w)

        return x

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b, _c, h, w)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b, c, h, w)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


class NonLocal(NonLocalModule):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
    """

    def __init__(self, inplanes, use_scale=True, **kwargs):
        planes = inplanes // 2
        self.use_scale = use_scale

        super(NonLocal, self).__init__(inplanes)
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=True)
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=True)
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1,
                           stride=1, bias=True)
        self.softmax = nn.Softmax(dim=2)
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1,
                           stride=1, bias=True)
        self.bn = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.size()

        t = t.view(b, c, -1).permute(0, 2, 1)
        p = p.view(b, c, -1)
        g = g.view(b, c, -1).permute(0, 2, 1)

        att = torch.bmm(t, p)

        if self.use_scale:
            att = att.div(c**0.5)

        att = self.softmax(att)
        x = torch.bmm(att, g)

        x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(b, c, h, w)

        x = self.z(x)
        x = self.bn(x) + residual

        return x


class DoubleAttention(NonLocalModule):

    def __init__(self, inplanes, **kwargs):
        super(DoubleAttention, self).__init__(inplanes)
        planes = inplanes // 4
        self.inter_channel = planes
        self.convA = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=1, bias=True)
        self.convB = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=1, bias=True)
        self.convV = nn.Conv2d(
            inplanes, planes, kernel_size=1, stride=1, bias=True)
        self.conv_expand = nn.Conv2d(
            planes, inplanes, kernel_size=1, stride=1, bias=True)
        self.bn = nn.BatchNorm2d(inplanes)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        b, c, h, w = x.size()

        A = self.convA(x)
        A = A.view(A.size()[0:2] + (-1,))
        B = self.convB(x)
        B = self.softmax(B.view(B.size()[0:2] + (-1,)))
        V = self.convV(x)
        V = self.softmax(V.view(V.size()[0:2] + (-1,)))

        Z = A.matmul(B.transpose(1, 2))  # (b,c,c)
        Z = Z.matmul(V)
        Z = Z.view(Z.size()[0:2] + (h, w))

        Z_out = self.conv_expand(Z)
        Z_out = self.bn(Z_out)
        return x + Z_out


class CCAttention(NonLocalModule):

    def __init__(self, inplanes, **kwargs):
        from .cc_attention import CrissCrossAttention
        super(CCAttention, self).__init__(inplanes)
        inter_channels = inplanes // 4
        self.conva = nn.Sequential(nn.Conv2d(inplanes, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels),
                                   nn.LeakyReLU())
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inplanes, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inplanes),
                                   nn.LeakyReLU())
        self.init_modules()

    def init_modules(self):
        _init_nl(self)
        nn.init.constant_(self.conva[-1].weight, 1)

    def forward(self, x):
        output = self.conva(x)
        for i in range(2):
            output = self.cca(output)
        output = self.convb(output)

        return x + output

if __name__ == '__main__':
    block_types = ['fft', 'nl', 'cgnl', 'a2', 'lo', 'fft_s', 'fft_p']
    in_channels = 32
    h = 28
    w = 24
    x = torch.rand(4, in_channels, h, w)
    for bt in block_types:
        block = get_nonlocal_block(bt)(
            in_channels, spatial_size=(h, w), spectrum_relu=True)
        out = block(x)
        print(bt, out.shape)
    block = get_nonlocal_block('fft')(
        in_channels, spectrum_relu=False, spatial_size=(h, w))
    out = block(x)
    print('fft', out.shape)
