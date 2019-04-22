'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torchvision import models

class vggPalmLocNet(nn.Module):
    def __init__(self, model):
        super(vggPalmLocNet, self).__init__()
        self.vggnet = nn.Sequential(*list(model.children())[:-1])
     #   for p in self.parameters():
     #       p.requires_grad = False
        self.outlinear = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            torch.nn.Dropout(0.5),  # drop 50% of the neuron
           # nn.BatchNorm1d(128),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            torch.nn.Dropout(0.5),
            nn.Linear(4096, 4)
        )

    def forward(self, x):
        x =self.vggnet(x)
        x = x.view(x.size(0), -1)
        output = self.outlinear(x)
        return output

def test():
    vgg = models.vgg16_bn(pretrained=False)
    net = vggPalmLocNet(vgg)
    print(net)
    x = torch.randn(2,3,224,224)
    y = net(x)
    print(y.size())

if __name__ == "__main__":
    test()
