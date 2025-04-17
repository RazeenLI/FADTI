import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDilatedConvolution_Old(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        residual_channels=32,
        dilation_channels=32,
        skip_channels=256,
        end_channels=512,
        kernel_size=2,
        blocks=4,
        layers=2,
    ):
        super().__init__()
        # device, num_nodes, dropout=0.3, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2
        self.blocks = blocks
        self.layers = layers

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))

        receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size),
                                                   dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size),
                                                 dilation=new_dilation))
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input):
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            filter = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filter * gate

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except:
                skip = 0
            skip = skip + s

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x



class GatedDilatedConvolution(nn.Module):
    def __init__(
            self,
            in_dim,              # 输入维度 = num_channels（embedding 之后）
            out_dim,             # 输出维度 = num_channels（embedding
            residual_channels=32,
            dilation_channels=32,
            skip_channels=256,
            end_channels=512,
            kernel_size=2,
            blocks=4,
            layers=2
        ):
        super().__init__()

        self.receptive_field = 1 + sum([2 ** i for i in range(layers)] * blocks)

        self.start_conv = nn.Conv1d(in_dim, residual_channels, kernel_size=1)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        for b in range(blocks):
            for i in range(layers):
                dilation = 2 ** i
                padding = (kernel_size - 1) * dilation

                self.filter_convs.append(nn.Conv1d(residual_channels, dilation_channels,
                                                   kernel_size, dilation=dilation, padding=padding))
                self.gate_convs.append(nn.Conv1d(residual_channels, dilation_channels,
                                                 kernel_size, dilation=dilation, padding=padding))
                self.residual_convs.append(nn.Conv1d(dilation_channels, residual_channels, kernel_size=1))
                self.skip_convs.append(nn.Conv1d(dilation_channels, skip_channels, kernel_size=1))
                self.bn.append(nn.BatchNorm1d(residual_channels))

        self.end_conv_1 = nn.Conv1d(skip_channels, end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(end_channels, out_dim, kernel_size=1)

    def forward(self, input):  # input: [B, C, T]
        in_len = input.size(2)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0))
        else:
            x = input

        x = self.start_conv(x)  # [B, R, T]
        skip = 0

        for i in range(len(self.filter_convs)):
            residual = x

            filter_out = torch.tanh(self.filter_convs[i](residual))
            gate_out = torch.sigmoid(self.gate_convs[i](residual))
            x = filter_out * gate_out

            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = skip + s

            x = self.residual_convs[i](x)
            x = x + residual[:, :, -x.size(2):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [B, out_dim, T]

        return x  # shape: [B, out_dim, T]
