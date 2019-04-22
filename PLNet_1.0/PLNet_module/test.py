from models.PLNet import vggPalmLocNet
from data.mydataset import MyDataset
from lossfuc.myloss import Myloss


############################
import cv2
import torch
import os
from torchvision import transforms, models
from torch.utils.data import DataLoader

######pic_size = 480
#######pic_resize =224

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if os.path.exists('./picture_total/testset/test.txt') :
    print('test.txt has been existed')
    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    test_data = MyDataset(txt='./picture_total/testset/test.txt', transform=transforms)
    test_loader = DataLoader(dataset=test_data, batch_size=10, num_workers=8)
else:
    print('you need to prepare your test.txt first!')

# 只加载训练好的参数
def test_PalmLocNet(test_x, test_y):
    vgg = models.vgg16_bn(pretrained=False)
    PLNet = vggPalmLocNet(vgg)
    PLNet.eval()
    if torch.cuda.is_available():
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load('./checkpoints/train_params_best.pth').items()})
    else:
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load('./checkpoints/train_params_best.pth',map_location='cpu').items()})
    PLNet.to(device)

    test_output_Plnet = PLNet(test_x)
    test_loss_func_Plnet = Myloss()
    test_GIoU_Plnet = test_loss_func_Plnet.MyGIoU(test_output_Plnet, test_y).sum()/test_y.shape[0]
    test_locMSEloss_Plnet = ((test_output_Plnet - test_y).pow(2).sum() / (4*test_y.shape[0]))
    test_loss_Plnet = test_loss_func_Plnet(test_output_Plnet, test_y)
    print('test GIoU: %.4f' % test_GIoU_Plnet, '\n'
          'test locMSEloss: %.4f' % test_locMSEloss_Plnet,'\n'
          'total test loss: %.4f' % test_loss_Plnet)
    return test_output_Plnet

def testpic(test_loader):
    for t, (tx, ty) in enumerate(test_loader):
        test_x = tx.to(device)
        test_y = ty.to(device) / 480

        oupt = test_PalmLocNet(test_x, test_y)
       # print('testx',test_x)
        oupt = 480 * oupt
        fh = open('./picture_total/testset/test.txt', 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append([words[0], words[1], words[2], words[3], words[4]])

        k = 0
        for p in imgs[t*10:(t+1)*10]:
            img = cv2.imread(p[0])
            # 画矩形框
            # 预测框
            cv2.rectangle(img, (oupt[k][0], oupt[k][1]), (oupt[k][2], oupt[k][3]), (0, 255, 0), 4)
            cv2.rectangle(img, (int(p[1]), int(p[2])), (int(p[3]), int(p[4])), (0, 0, 255), 4)
            cv2.imwrite('./picture_total/testset/testtruth/'+ str(t) + str(k) + '_test_truth.jpg', img)
            k += 1

# 运行训练以及测试模型
if __name__ == '__main__':
    testpic(test_loader)
