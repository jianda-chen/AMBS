# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, rew_length, num_layers=2, num_filters=32, stride=None):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.rew_length = rew_length
        self.dyn_length = feature_dim - rew_length
        self.ratio =  nn.Parameter(torch.tensor([0., 0.]))
        # self.ratio =  nn.Parameter(torch.tensor([0., 0.]), requires_grad=False)
        # self.ratio =  nn.Parameter(torch.tensor([-1000., 1000.]), requires_grad=False)
        # self.softmax_ratio = nn.Parameter(torch.tensor([1., 0.99]), requires_grad=False)
        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False, pre_ratio=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        pre_ratio_out = out
        
        softmax_ratio = torch.softmax(self.ratio, dim=-1)
        # softmax_ratio = self.softmax_ratio
        expand_softmax_ratio = torch.repeat_interleave(
            softmax_ratio, torch.tensor((self.rew_length, self.dyn_length)).to(softmax_ratio.device))
        out = out * expand_softmax_ratio[None, ...]

        self.outputs['ln'] = out

        if pre_ratio:
            return out, pre_ratio_out
        else:
            return out
    # def forward(self, obs, detach=False, pre_ratio=False):
    #     h = self.forward_conv(obs)

    #     if detach:
    #         h = h.detach()

    #     h_fc = self.fc(h)
    #     self.outputs['fc'] = h_fc

    #     out = self.ln(h_fc)
    #     if detach:
    #         out = out.detach()
    #     pre_ratio_out = out
        
    #     softmax_ratio = torch.softmax(self.ratio, dim=-1)
    #     # softmax_ratio = self.softmax_ratio
    #     expand_softmax_ratio = torch.repeat_interleave(
    #         softmax_ratio, torch.tensor((self.rew_length, self.dyn_length)).to(softmax_ratio.device))
    #     out = out * expand_softmax_ratio[None, ...]

    #     self.outputs['ln'] = out

    #     if detach:
    #         pre_ratio_out = pre_ratio_out.detach()
    #         out = out.detach()

    #     if pre_ratio:
    #         return out, pre_ratio_out
    #     else:
    #         return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])
    # def copy_conv_weights_from(self, source):
    #     """Tie convolutional layers"""
    #     # only tie conv layers
    #     for i in range(self.num_layers):
    #         tie_weights(src=source.convs[i], trg=self.convs[i])
    #     tie_weights(src=source.fc, trg=self.fc)
    #     tie_weights(src=source.ln, trg=self.ln)
    #     self.ratio = source.ratio


    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderCarla096(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, rew_length, num_layers=2, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))

        out_dims = 100 #100 # 16 # if defaults change, adjust this as needed
        self.fc = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.rew_length = rew_length
        self.dyn_length = feature_dim - rew_length
        self.ratio =  nn.Parameter(torch.tensor([0., 0.]))

        self.outputs = dict()


class PixelEncoderCarla098(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, rew_length, num_layers=4, num_filters=32, stride=1):
        super(PixelEncoder, self).__init__()
        print("PixelEncoderCarla098")

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        # self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        # self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        # self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        # self.convs.append(nn.Conv2d(256, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 64, 3, stride=2))
        self.convs.append(nn.Conv2d(64, 64, 3, stride=2))
        self.convs.append(nn.Conv2d(64, 64, 3, stride=2))

        # out_dims = 56  # 3 cameras
        # out_dims = 100  # 5 cameras
        out_dims = 16
        self.fc = nn.Linear(64 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.rew_length = rew_length
        self.dyn_length = feature_dim - rew_length
        self.ratio =  nn.Parameter(torch.tensor([0., 0.]))

        self.outputs = dict()


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixelCarla096': PixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                       'identity': IdentityEncoder}


def make_encoder(
    encoder_type, obs_shape, feature_dim, rew_length, num_layers, num_filters, stride
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, rew_length, num_layers, num_filters, stride
    )
