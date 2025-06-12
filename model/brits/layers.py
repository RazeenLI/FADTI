import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import math
import utils
import argparse
# import data_loader

from model.brits import rits
from sklearn import metrics

# from ipdb import set_trace

SEQ_LEN = 48
RNN_HID_SIZE = 64


class BackboneBRITS(nn.Module):
    def __init__(
            self, 
            n_steps: int,
            n_features: int,
            rnn_hidden_size: int,
        ):
        super(BackboneBRITS, self).__init__()

        # data settings
        self.n_steps = n_steps
        self.n_features = n_features
        # imputer settings
        self.rnn_hidden_size = rnn_hidden_size

        self.rits_f = rits.RITS(n_steps, n_features, rnn_hidden_size)
        self.rits_b = rits.RITS(n_steps, n_features, rnn_hidden_size)

    # def forward(self, data): 
    def forward(self, forward_data, backword_data):
        # ret_f = self.rits_f(data, 'forward')
        # ret_b = self.reverse(self.rits_b(data, 'backward'))
        (
            f_imputed_data,
            f_reconstruction,
            f_hidden_states,
            f_reconstruction_loss,
        ) = self.rits_f(forward_data)
        (
            b_imputed_data,
            b_reconstruction,
            b_hidden_states,
            b_reconstruction_loss,
        ) = self.reverse(self.rits_b(backword_data))

        imputed_data = (f_imputed_data + b_imputed_data) / 2
        consistency_loss = self.get_consistency_loss(f_imputed_data, b_imputed_data)
        reconstruction_loss = f_reconstruction_loss + b_reconstruction_loss

        return (
            imputed_data,
            f_reconstruction,
            b_reconstruction,
            f_hidden_states,
            b_hidden_states,
            consistency_loss,
            reconstruction_loss,
        )

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = torch.tensor(indices, dtype=torch.long, device=tensor_.device, requires_grad=False)
            return tensor_.index_select(1, indices)

            # indices = Variable(torch.LongTensor(indices), requires_grad = False)

            # if torch.cuda.is_available():
            #     indices = indices.cuda()

            # return tensor_.index_select(1, indices)
        collector = []
        for value in ret:
            collector.append(reverse_tensor(value))

        return tuple(collector)

        # for key in ret:
        #     ret[key] = reverse_tensor(ret[key])

        # return ret

    # def run_on_batch(self, data, optimizer, epoch=None):
    #     ret = self(data)

    #     if optimizer is not None:
    #         optimizer.zero_grad()
    #         ret['loss'].backward()
    #         optimizer.step()

    #     return ret
