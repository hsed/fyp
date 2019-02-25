import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, z_dim, hand_side_invariance=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        if hand_side_invariance:
            self.fc = nn.Linear(512 * 2 * 2 + 2, z_dim*2)
        else:
            self.fc = nn.Linear(512 * 2 * 2, z_dim*2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, hand_side=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if hand_side is None:
            x = self.fc(x)
        else:
            x = self.fc(torch.cat((x, hand_side), dim=1))

        return x


def resnet18(z_dim, hand_side_invariance):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], z_dim, hand_side_invariance)

    return model

def resnet34(z_dim, hand_side_invariance):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], z_dim, hand_side_invariance)

    return model



class rgb_encoder(nn.Module):
    def __init__(self, z_dim, hand_side_invariance=False):
        super(rgb_encoder, self).__init__()

        # IN: 3 x 128 x 128
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(), # 16 x 64 x 64
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(), # 32 x 32 x 32
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(), # 32 x 32 x 32
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # 64 x 16 x 16
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # 64 x 16 x 16
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(), # 128 x 8 x 8
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(), # 128 x 8 x 8
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # 256 x 4 x 4
        )
        self.conv_out_dim = 256 * 4 * 4
        if hand_side_invariance:
            self.lin_lay = nn.Linear(self.conv_out_dim + 2, 2 * z_dim)
        else:
            self.lin_lay = nn.Linear(self.conv_out_dim, 2 * z_dim)
        # self.conv_blocks = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 2, 1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(), # 16 x 64 x 64
        #     nn.Conv2d(16, 32, 3, 2, 1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(), # 32 x 32 x 32
        #     nn.Conv2d(32, 64, 3, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(), # 64 x 16 x 16
        #     nn.Conv2d(64, 128, 3, 2, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(), # 128 x 8 x 8
        # )
        # self.lin_lay = nn.Linear(128 * 8 * 8, 2 * z_dim)

    def forward(self, x, hand_side=None):
        out_conv = self.conv_blocks(x)
        # in_lay = out_conv.view(-1, 128 * 8 * 8)
        in_lay = out_conv.view(-1, self.conv_out_dim)
        if hand_side is None:
            out_lay = self.lin_lay(in_lay)
        else:
            out_lay = self.lin_lay(torch.cat((in_lay, hand_side), dim=1))
        return out_lay

class rgb_decoder(nn.Module):
    def __init__(self, z_dim):
        super(rgb_decoder, self).__init__()
        self.lin_lay = nn.Sequential(
            nn.Linear(z_dim, 128 * 8 * 8),
            nn.BatchNorm1d(128 * 8 * 8),
            nn.ReLU()
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # 64 x 16 x 16
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(), # 32 x 32 x 32
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(), # 16 x 64 x 64
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
        )
        # OUT: 3 x 128 x 128

    def forward(self, x):
        out_lay = self.lin_lay(x)
        in_conv = out_lay.view(-1, 128, 8, 8)
        out_conv = self.conv_blocks(in_conv)
        return out_conv


class joints_encoder(nn.Module):
    def __init__(self, z_dim, in_dim):
        super(joints_encoder, self).__init__()
        in_dim = torch.IntTensor(in_dim)
        self.in_size = in_dim.prod()
        self.lin_lays = nn.Sequential(
            nn.Linear(self.in_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * z_dim)
        )

    def forward(self, x):
        return self.lin_lays(x.view(-1, self.in_size))

class joints_decoder(nn.Module):
    def __init__(self, z_dim, in_dim):
        super(joints_decoder, self).__init__()
        self.in_dim = torch.IntTensor(in_dim)
        self.in_size = self.in_dim.prod()
        self.lin_lays = nn.Sequential(
            nn.Linear(z_dim, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, self.in_size)
        )

    def forward(self, x):
        out_lays = self.lin_lays(x)
        return out_lays.view(-1, self.in_dim[0], self.in_dim[1])


class VAE(nn.Module):
    """Variational Autoencoder module that allows cross-embedding

    Arguments:
        in_dim(list): Input dimension.
        z_dim(int): Noise dimension
        encoder(nn.Module): The encoder module. Its output dim must be 2*z_dim
        decoder(nn.Module): The decoder module. Its output dim must be in_dim.prod()
    """
    def __init__(self, z_dim, encoder, decoder):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = encoder
        self.decoder = decoder


    def encode(self, x, hand_side=None):
        h_i = self.encoder(x, hand_side)
        # Split to mu and logvar
        return h_i[:, self.z_dim:], h_i[:, :self.z_dim]

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def forward(self, x, vae_decoder=None, hand_side=None):
        mu, logvar = self.encode(x, hand_side)
        z = self.reparameterize(mu, logvar)

        # If no seperate decoder is specified, use own.
        if not vae_decoder:
            dec = self.decoder
        else:
            dec = vae_decoder.decoder

        return dec(z), mu, logvar
