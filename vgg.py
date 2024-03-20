import torch
import torch.nn as nn
from torch.autograd import Variable
import math  # init


class vgg(nn.Module):

    def __init__(self, dataset='cifar10', init_weights=True, cfg=None):
        """
            :param dataset: cifar10 or cifar100
            :param init_weights: init weights
            :param cfg: configuration of vgg, None for default configuration
        """
        super(vgg, self).__init__()
        if cfg is None:
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        self.feature = self.make_layers(cfg, True) # feature extractor

        if dataset == 'cifar100':
            num_classes = 100
        elif dataset == 'cifar10':
            num_classes = 10
        self.classifier = nn.Linear(cfg[-1], num_classes) # classifier (fully connected layers)
        if init_weights:
            self._initialize_weights()

    def make_layers(self, cfg, batch_norm=False):
        """
            :param cfg: configuration of vgg
            :param batch_norm: batch normalization
            :return: sequential layers of vgg
        """
        layers = []
        in_channels = 3 # input channel
        for v in cfg: # loop over the configuration
            if v == 'M': # max pooling
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else: # convolutional layer
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm: # batch normalization
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else: # without batch normalization
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        """
            :param x: input i.e. image
            :return: output i.e. class scores
        """
        x = self.feature(x) # feature extractor
        x = nn.AvgPool2d(2)(x) # average pooling
        x = x.view(x.size(0), -1) # flatten
        y = self.classifier(x) # classifier
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # convolutional layer
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d): # batch normalization
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear): # fully connected layer
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    net = vgg()
    x = Variable(torch.FloatTensor(16, 3, 40, 40)) # batch size, input channel, input height, input width
    y = net(x)
    print(y.data.shape)
