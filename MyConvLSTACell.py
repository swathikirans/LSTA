import torch
import torch.nn as nn
import torch.nn.functional as F


class MyConvLSTACell(nn.Module):
    def __init__(self, input_size, memory_size, c_cam_classes=100, kernel_size=3,
                 stride=1, padding=1, zero_init=False):
        super(MyConvLSTACell, self).__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.c_classifier = nn.Linear(memory_size, c_cam_classes, bias=False)
        self.coupling_fc = nn.Linear(memory_size, c_cam_classes, bias=False)
        self.avgpool = nn.AvgPool2d(7)

        # Attention params

        self.conv_i_s = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_i_cam = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.conv_f_s = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_cam = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.conv_a_s = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_a_cam = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        self.conv_o_s = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_cam = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

        if zero_init:
            torch.nn.init.constant(self.conv_i_s.weight, 0)
            torch.nn.init.constant(self.conv_i_s.bias, 0)
            torch.nn.init.constant(self.conv_i_cam.weight, 0)

            torch.nn.init.constant(self.conv_f_s.weight, 0)
            torch.nn.init.constant(self.conv_f_s.bias, 0)
            torch.nn.init.constant(self.conv_f_cam.weight, 0)

            torch.nn.init.constant(self.conv_a_s.weight, 0)
            torch.nn.init.constant(self.conv_a_s.bias, 0)

            torch.nn.init.constant(self.conv_o_s.weight, 0)
            torch.nn.init.constant(self.conv_o_s.bias, 0)
            torch.nn.init.constant(self.conv_o_cam.weight, 0)
        else:
            torch.nn.init.xavier_normal(self.conv_i_s.weight)
            torch.nn.init.constant(self.conv_i_s.bias, 0)
            torch.nn.init.xavier_normal(self.conv_i_cam.weight)

            torch.nn.init.xavier_normal(self.conv_f_s.weight)
            torch.nn.init.constant(self.conv_f_s.bias, 0)
            torch.nn.init.xavier_normal(self.conv_f_cam.weight)

            torch.nn.init.xavier_normal(self.conv_a_s.weight)
            torch.nn.init.constant(self.conv_a_s.bias, 0)
            torch.nn.init.xavier_normal(self.conv_a_cam.weight)

            torch.nn.init.xavier_normal(self.conv_o_s.weight)
            torch.nn.init.constant(self.conv_o_s.bias, 0)
            torch.nn.init.xavier_normal(self.conv_o_cam.weight)

        # Memory params

        self.conv_i_x = nn.Conv2d(input_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_i_c = nn.Conv2d(memory_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        self.conv_f_x = nn.Conv2d(input_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_f_c = nn.Conv2d(memory_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        self.conv_c_x = nn.Conv2d(input_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_c_c = nn.Conv2d(memory_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        self.conv_o_x = nn.Conv2d(input_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv_o_c = nn.Conv2d(memory_size, memory_size, kernel_size=kernel_size, stride=stride, padding=padding,
                                  bias=False)

        if zero_init:
            torch.nn.init.constant(self.conv_i_x.weight, 0)
            torch.nn.init.constant(self.conv_i_x.bias, 0)
            torch.nn.init.constant(self.conv_i_c.weight, 0)

            torch.nn.init.constant(self.conv_f_x.weight, 0)
            torch.nn.init.constant(self.conv_f_x.bias, 0)
            torch.nn.init.constant(self.conv_f_c.weight, 0)

            torch.nn.init.constant(self.conv_c_x.weight, 0)
            torch.nn.init.constant(self.conv_c_x.bias, 0)
            torch.nn.init.constant(self.conv_c_c.weight, 0)

            torch.nn.init.constant(self.conv_o_x.weight, 0)
            torch.nn.init.constant(self.conv_o_x.bias, 0)
            torch.nn.init.constant(self.conv_o_c.weight, 0)
        else:
            torch.nn.init.xavier_normal(self.conv_i_x.weight)
            torch.nn.init.constant(self.conv_i_x.bias, 0)
            torch.nn.init.xavier_normal(self.conv_i_c.weight)

            torch.nn.init.xavier_normal(self.conv_f_x.weight)
            torch.nn.init.constant(self.conv_f_x.bias, 0)
            torch.nn.init.xavier_normal(self.conv_f_c.weight)

            torch.nn.init.xavier_normal(self.conv_c_x.weight)
            torch.nn.init.constant(self.conv_c_x.bias, 0)
            torch.nn.init.xavier_normal(self.conv_c_c.weight)

            torch.nn.init.xavier_normal(self.conv_o_x.weight)
            torch.nn.init.constant(self.conv_o_x.bias, 0)
            torch.nn.init.xavier_normal(self.conv_o_c.weight)

    def forward(self, x, cam, state_att, state_inp, x_flow_i=0, x_flow_f=0, x_flow_c=0, x_flow_o=0):
        # state_att = [a, s]
        # state_inp = [atanh(c), o]

        a_t_1 = state_att[0]
        s_t_1 = state_att[1]

        c_t_1 = F.tanh(state_inp[0])
        o_t_1 = state_inp[1]

        # Attention recurrence

        i_s = F.sigmoid(self.conv_i_s(s_t_1) + self.conv_i_cam(cam))
        f_s = F.sigmoid(self.conv_f_s(s_t_1) + self.conv_f_cam(cam))
        o_s = F.sigmoid(self.conv_o_s(s_t_1) + self.conv_o_cam(cam))
        a_tilde = F.tanh(self.conv_a_s(s_t_1) + self.conv_a_cam(cam))
        a = (f_s * a_t_1) + (i_s * a_tilde)
        s = o_s * F.tanh(a)
        u = s + cam  # hidden state + cam

        u = F.softmax(u.view(u.size(0), -1), 1)
        u = u.view(u.size(0), 1, 7, 7)

        x_att = x * u.expand_as(x)

        i_x = F.sigmoid(self.conv_i_c(o_t_1 * c_t_1) + self.conv_i_x(x_att) + x_flow_i)
        f_x = F.sigmoid(self.conv_f_c(o_t_1 * c_t_1) + self.conv_f_x(x_att) + x_flow_f)
        c_tilde = F.tanh(self.conv_c_c(o_t_1 * c_t_1) + self.conv_c_x(x_att) + x_flow_c)
        c = (f_x * state_inp[0]) + (i_x * c_tilde)

        c_vec = self.avgpool(c).view(c.size(0), -1)
        c_logits = self.c_classifier(c_vec) + self.coupling_fc(self.avgpool(x_att).view(x_att.size(0), -1))
        c_probs, c_idxs = c_logits.sort(1, True)
        c_class_idx = c_idxs[:, 0]
        c_cam = self.c_classifier.weight[c_class_idx].unsqueeze(2).unsqueeze(2) * c
        o_x = F.sigmoid(self.conv_o_x(o_t_1 * c_t_1) + self.conv_o_c(c_cam)) 

        state_att = [a, s]
        state_inp = [c, o_x]
        return state_att, state_inp, x_att