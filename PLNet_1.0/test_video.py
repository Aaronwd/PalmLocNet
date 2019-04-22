import numpy as np
import cv2
from torchvision import models
import torch
from PIL import Image
import torchvision

from PLNet_module.models.PLNet import vggPalmLocNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def testvideolocnet():
    vgg = models.vgg16_bn(pretrained=False)
    PLNet = vggPalmLocNet(vgg)
    PLNet.eval()
    if torch.cuda.is_available():
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load( '.../train_params_best.pth').items()})
    else:
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load('.../train_params_best.pth',map_location='cpu').items()})
    PLNet.to(device)

    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame1 = torchvision.transforms.Resize(224)(frame)
            tframe = torchvision.transforms.ToTensor()(frame1)
            tframe = tframe.unsqueeze(0)
            print('tframe', tframe.shape)

            outloc = PLNet(tframe.to(device))
            xx = 640
            yy = 480
            frame = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (xx*outloc[0][0], yy*outloc[0][1]), (xx*outloc[0][2], yy*outloc[0][3]), (0, 255, 0), 4)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    testvideolocnet()
