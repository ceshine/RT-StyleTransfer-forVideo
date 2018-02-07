import torch
from style_network import *
from loss_network import *


class Transfer:
    def __init__(self, data_path, vgg_path, lr, spatial_a, spatial_b, spatial_r, temporal_lambda):
        self.data_path = data_path
        self.lr = lr

        self.s_a = spatial_a
        self.s_b = spatial_b
        self.s_r = spatial_r 
        self.t_l = temporal_lambda

        self.style_net = StyleNet()
        self.loss_net = LossNet(vgg_path)

    def train(self):
