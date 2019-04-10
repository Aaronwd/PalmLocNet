import torch
import torch.nn as nn
import argparse
import torch.utils.data as data
import os
from tensorboardX import SummaryWriter

#为处理数据引入的模块
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#画boundingbox模块
import numpy as np
import cv2

import torch.nn.functional as F

######pic_size = 480
#######pic_resize =128

#设置超参数
parser = argparse.ArgumentParser(description='super params')
parser.add_argument('-e','--EPOCH', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('-b','--BATCH_SIZE', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-l','--LR', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('-m','--MODELFOLDER',type= str, default='./modelvggorg/',
                help="folder to store model")
# 有点小问题，要保证和实际的数据集的路径保持一致，不够智能
parser.add_argument('-p','--PICTUREFOLDER',type= str, default='./picture/',
                help="folder to store trained picture")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1','True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0','False'):
        return False
    else: raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser.add_argument('-t','--TrainOrNot', type=str2bool, nargs='?', const=True, required= True,
                    help='TrainOrNot (default: True)')
parser.add_argument('-v','--VideotestOrNot', type=str2bool, nargs='?', const=True, required= True,
                    help='VideotestOrNot (default: True)')
args = parser.parse_args()

#检测是否有利用的gpu环境
# use_gpu = torch.cuda.is_available()
# print('use GPU:',use_gpu)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#将数据导入到pytorch的dataloader中，以便进行批训练、打乱顺序等动作
def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], torch.tensor([float(words[1]),float(words[2]),float(words[3]),float(words[4])])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

test_batch_size = 10

if os.path.exists(args.PICTUREFOLDER+'trainset/'+'train.txt') and os.path.exists(args.PICTUREFOLDER+'testset/'+'test.txt') :
    print('train.txt and test.txt have been existed')
    transforms = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
   #     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    train_data = MyDataset(txt=args.PICTUREFOLDER + 'trainset/' + 'train.txt', transform=transforms)
    test_data = MyDataset(txt=args.PICTUREFOLDER + 'testset/' + 'test.txt', transform=transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=args.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size)
else:
    print('you need to prepare your train.txt and test.txt first!')

#测试数据
for k, (tx, ty) in enumerate(test_loader):
    test_x = tx.to(device)
    test_y = ty.to(device)/480

#神经网络建模
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.outlinear = nn.Sequential(
            nn.Linear(512 * 4 * 4, 128),
            torch.nn.Dropout(0.5),  # drop 50% of the neuron
            #   nn.BatchNorm1d(6400),
            nn.ReLU(),
            nn.Linear(128, 4),
            # torch.nn.Dropout(0.5),  # drop 50% of the neuron
            # #    nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        output = self.outlinear(x)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def PalmLocNet():
    return VGG('VGG16')

#自定义损失函数
class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
    def MyGIoU(self, pred_loc, truth_loc):
        pred_loc = pred_loc.to(device)
        truth_loc = truth_loc.to(device)
        # if use_gpu:
        #     pred_loc = pred_loc.cuda()
        #     truth_loc = truth_loc.cuda()
        # else:
        #     pass
        x_p1 = pred_loc[:,0]
        y_p1 = pred_loc[:,1]
        x_p2 = pred_loc[:,2]
        y_p2 = pred_loc[:,3]
        #print('x_p1, y_p1, x_p2, y_p2',x_p1, y_p1, x_p2, y_p2)

        x_g1 = truth_loc[:,0]
        y_g1 = truth_loc[:,1]
        x_g2 = truth_loc[:,2]
        y_g2 = truth_loc[:,3]
        #print(x_g1, y_g1, x_g2, y_g2)

        A_g = (x_g2 - x_g1) * (y_g2 - y_g1)
        #防止xp2小于xp1
        zer = torch.zeros(x_g1.shape).to(device)
        # if use_gpu:
        #     zer = torch.zeros(x_g1.shape).cuda()
        # else:
        #     zer = torch.zeros(x_g1.shape)
        A_p = (torch.max((x_p2 - x_p1), zer)) * (torch.max((y_p2 - y_p1), zer))
        #print(A_g, A_p)

        x_I1 = torch.max(x_p1, x_g1)
        x_I2 = torch.min(x_p2, x_g2)
        y_I1 = torch.max(y_p1, y_g1)
        y_I2 = torch.min(y_p2, y_g2)

        I = (torch.max((x_I2 - x_I1), zer)) * (torch.max((y_I2 - y_I1), zer))
        # print(I)

        x_C1 = torch.min(x_p1, x_g1)
        x_C2 = torch.max(x_p2, x_g2)
        y_C1 = torch.min(y_p1, y_g1)
        y_C2 = torch.max(y_p2, y_g2)
        #print(x_C1, x_C2, y_C1, y_C2)

        A_c = (x_C2 - x_C1) * (y_C2 - y_C1)
        U = A_g + A_p - I
        myIoU = I / U
        myGIoU = myIoU - (A_c - U) / A_c
        #print('IoU: %.4f, gIoU: %.4f' % (myIoU, myGIoU))
        return myGIoU

    def forward(self, pred_loc, truth_loc):
        pred_loc = pred_loc.to(device)
        truth_loc = truth_loc.to(device)
        locMSEloss = ((pred_loc-truth_loc).pow(2).sum()/(4*truth_loc.shape[0]))
        gIoUloss = 1- self.MyGIoU(pred_loc, truth_loc).sum()/truth_loc.shape[0]
        myloss = locMSEloss+gIoUloss
        return myloss


#训练以及保存模型数据
def train_PalmLocNet(train_loader, test_x, test_y):
    if os.path.exists(args.MODELFOLDER + 'train_params_best.pth'):
        print('reload the last best model parameters')
        pal = PalmLocNet()
     #   pal.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth'))
        pal.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.MODELFOLDER + 'train_params_best.pth').items()})
        if torch.cuda.device_count() > 1:
            print('lets use', torch.cuda.device_count(), 'GPUs')
            palnet = nn.DataParallel(pal)
        palnet.to(device)

    else:
        print('It is the first time to train the model!')
        pal = PalmLocNet()
        if torch.cuda.device_count() > 1:
            print('lets use', torch.cuda.device_count(), 'GPUs')
            palnet = nn.DataParallel(pal)
        palnet.to(device)

    optimizer = torch.optim.Adam(palnet.parameters(),lr= args.LR)
    loss_func = Myloss()

    for epoch in range(args.EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = x.to(device)
            b_y = y.to(device)/480

          #  print('b_x', b_x)
          #  print('b_y', b_y)
          #  print('b_x.shape', b_x.shape)
          #  print('b_y.shape', b_y.shape)
            output = palnet(b_x)
          #  print('output', output)
          #  output = 480*output
            loss = loss_func(output, b_y)
            optimizer.zero_grad()  # 将上一步梯度值清零
            loss.backward()  # 求此刻各参数的梯度值
            optimizer.step()  # 更新参数

            # 可视化模型结构
            # with SummaryWriter(log_dir='PLNet01') as w:
            #     w.add_graph(palnet,(b_x,))
            #     w.add_scalar('Train', loss, global_step=(epoch+1)*100+step)

            if step % 100 == 0:
                palnet.eval()
                test_output = palnet(test_x)
                print('test_output[0]',test_output[0])
                print('test_y[0]', test_y[0])
               # test_output = 480 * test_output
               # print('480 * test_output',test_output)
                test_loss_func = Myloss()
                test_GIoU = test_loss_func.MyGIoU(test_output,test_y).sum() /(test_y.shape[0])
                test_locMSEloss = ((test_output - test_y).pow(2).sum() /(4*test_y.shape[0]))
                test_loss = test_loss_func(test_output,test_y)
                print('Epoch', epoch, '\n'
                      'train loss: %.4f' % loss.data.cpu().numpy(),'\n'
                      'test GIoU: %.4f' % test_GIoU,'\n'
                      'test locMSEloss: %.4f' % test_locMSEloss,'\n'
                      'total test loss: %.4f' % test_loss)
                palnet.train()
                

        # 检查是否有模型文件夹，没有就自行创建一个
        if not os.path.isdir(args.MODELFOLDER):
            os.makedirs(args.MODELFOLDER)

        #保存训练loss最小的模型参数（按周期记步）
        if epoch == 0:
            if os.path.exists(args.MODELFOLDER + 'train_params_best.pth'):
                print('exist the train_params_best.pth!')
                pass
            else:
                print('first make the train_params_best.pth')
                torch.save(palnet.state_dict(), args.MODELFOLDER + 'train_params_best.pth')
            best_loss = loss
            print('best_loss in epoch 0:', best_loss)
           # print('compare_loss:', loss)
        else:
           # print('compare_loss.append:', compare_loss)
            if loss < best_loss:
                torch.save(palnet.state_dict(), args.MODELFOLDER + 'train_params_best.pth')
                print('save the best trained model in epoch', epoch)
                best_loss = loss
                print('new best_loss:',best_loss)
            else:
                print('no better in this epoch', epoch)

#只加载训练好的参数
def test_PalmLocNet(test_x, test_y):
    PLNet = PalmLocNet()
    PLNet.eval()
    if torch.cuda.is_available():
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(args.MODELFOLDER + 'train_params_best.pth').items()})
    else:
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(args.MODELFOLDER + 'train_params_best.pth',map_location='cpu').items()})
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
        oupt = 480 * oupt
        fh = open(args.PICTUREFOLDER + 'testset/' + 'test.txt', 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append([words[0], words[1], words[2], words[3], words[4]])

        k = 0
        for p in imgs[t*test_batch_size:(t+1)*test_batch_size]:
            img = cv2.imread(p[0])
            # 画矩形框
            # 预测框
            cv2.rectangle(img, (oupt[k][0], oupt[k][1]), (oupt[k][2], oupt[k][3]), (0, 255, 0), 4)
            cv2.rectangle(img, (int(p[1]), int(p[2])), (int(p[3]), int(p[4])), (0, 0, 255), 4)
            cv2.imwrite(args.PICTUREFOLDER + 'testset/' + 'testtruth/'+ str(t) + str(k) + '_test_truth.jpg', img)
            k += 1

def testvideolocnet():
    PLNet = PalmLocNet()
    PLNet.eval()
    if torch.cuda.is_available():
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(args.MODELFOLDER + 'train_params_best.pth').items()})
    else:
        PLNet.load_state_dict(
            {k.replace('module.', ''): v for k, v in torch.load(args.MODELFOLDER + 'train_params_best.pth',map_location='cpu').items()})
   # PLNet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth'))
  #  PLNet.load_state_dict(
  #      {k.replace('module.', ''): v for k, v in torch.load(args.MODELFOLDER + 'train_params_best.pth').items()})
    PLNet.to(device)
    # if use_gpu:
    #     PLNet = PalmLocNet()
    #     PLNet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth'))
    #     PLNet = PLNet.cuda()
    # else:
    #     PLNet = PalmLocNet()
    #     PLNet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth',map_location='cpu'))
    cap = cv2.VideoCapture(1)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            print(frame.shape)
            frame = cv2.resize(frame, (480, 480))
            frame1 = cv2.resize(frame, (128, 128))
             #用permute改变高维tensor的维数位置
            tframe = torch.from_numpy(frame1).permute(2,0,1)
             #加一维
            tframe = tframe.unsqueeze(0)
            tframe = tframe.float()
            outloc = PLNet(tframe)
            print(outloc)
            outloc = 480*outloc
            print(outloc)
         #   cv2.rectangle(frame, (1, 60), (100, 200), (0, 255, 0), 4)
            cv2.rectangle(frame, (outloc[0][0], outloc[0][1]), (outloc[0][2], outloc[0][3]), (0, 255, 0), 4)
          #  frame = cv2.resize(frame, (640, 480))
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
                break
    cap.release()
    cv2.destroyAllWindows()


#运行训练以及测试模型
if (__name__ == '__main__') and args.TrainOrNot:
    train_PalmLocNet(train_loader, test_x, test_y)

if (__name__ == '__main__') and (not args.TrainOrNot):
    if args.VideotestOrNot:
        testvideolocnet()
    else:
        testpic(test_loader)







