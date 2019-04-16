import torch
import torch.nn as nn
import cv2
from torchvision import transforms, models
import os

#定义神经网络（使用了pytorch已有的网络，部分参数训练时固定）
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test(testpath):

    vgg = models.vgg16_bn(pretrained=False)
    PLNet = vggPalmLocNet(vgg)
    PLNet.eval()
    if torch.cuda.is_available():
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load('./prevgg_param_class_fn2_g/train_params_best.pth').items()})
    else:
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load('./prevgg_param_class_fn2_g/train_params_best.pth',map_location='cpu').items()})
    PLNet.to(device)

    os.chdir(testpath)
    testfilenames = os.listdir()
   # print(testfilenames)
    for i in testfilenames:
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break
        imgpath = testpath+i
        print(imgpath)
        frame = cv2.imread(imgpath)
        frame = cv2.resize(frame, (480, 480))
        frame1 = cv2.resize(frame, (224, 224))
        # 用permute改变高维tensor的维数位置
        tframe = torch.from_numpy(frame1).permute(2, 0, 1)
        # 加一维
        tframe = tframe.unsqueeze(0)
        tframe = tframe.float()
        outloc = PLNet(tframe)
      #  print(outloc)
        outloc = 480 * outloc
        #print(outloc)
        #   cv2.rectangle(frame, (1, 60), (100, 200), (0, 255, 0), 4)
        cv2.rectangle(frame, (outloc[0][0], outloc[0][1]), (outloc[0][2], outloc[0][3]), (0, 255, 0), 4)
        #  frame = cv2.resize(frame, (640, 480))
        cv2.imshow('frame', frame)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    test('/home/aaron/桌面/github-wd/video-pic-test/shi/')
