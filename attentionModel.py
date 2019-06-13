import resNetNew
from torch.autograd import Variable
from MyConvLSTACell import *


class attentionModel(nn.Module):
    def __init__(self, num_classes=51, mem_size=512, c_cam_classes=1000):
        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resNetNew.resnet34(True, True)
        self.mem_size = mem_size
        self.lsta_cell = MyConvLSTACell(512, mem_size, c_cam_classes)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)

    def forward(self, inputVariable):
        state_att = (Variable(torch.zeros(inputVariable.size(1), 1, 7, 7).cuda()),
                     Variable(torch.zeros(inputVariable.size(1), 1, 7, 7).cuda()))
        state_inp = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        for t in range(inputVariable.size(0)):
            logit, feature_conv, x = self.resNet(inputVariable[t])
            bz, nc, h, w = feature_conv.size()
            feature_conv1 = feature_conv.view(bz, nc, h * w)
            probs, idxs = logit.sort(1, True)
            class_idx = idxs[:, 0]
            cam = torch.bmm(self.resNet.fc.weight[class_idx].unsqueeze(1), feature_conv1).view(x.size(0), 1, 7, 7)
            state_att, state_inp, _ = self.lsta_cell(x, cam, state_att, state_inp)
        feats = self.avgpool(state_inp[0]).view(state_inp[0].size(0), -1)
        logits = self.classifier(feats)
        return logits, feats
