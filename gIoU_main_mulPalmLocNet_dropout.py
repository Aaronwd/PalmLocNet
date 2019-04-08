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

######pic_size = 480

#设置超参数
parser = argparse.ArgumentParser(description='super params')
parser.add_argument('-e','--EPOCH', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('-b','--BATCH_SIZE', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-l','--LR', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('-m','--MODELFOLDER',type= str, default='./model01/',
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
use_gpu = torch.cuda.is_available()
print('use GPU:',use_gpu)

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

if os.path.exists(args.PICTUREFOLDER+'trainset/'+'train.txt') and os.path.exists(args.PICTUREFOLDER+'testset/'+'test.txt') :
    print('train.txt and test.txt have been existed')
    train_data = MyDataset(txt=args.PICTUREFOLDER + 'trainset/' + 'train.txt', transform=transforms.ToTensor())
    test_data = MyDataset(txt=args.PICTUREFOLDER + 'testset/' + 'test.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_data, batch_size=args.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=10)
else:
    print('you need to prepare your train.txt and test.txt first!')

#测试数据
for k, (tx, ty) in enumerate(test_loader):
    if use_gpu:
        test_x = tx.cuda()
        test_y = ty.cuda()
    else:
        test_x = tx
        test_y = ty

#神经网络建模
class PalmLocNet(nn.Module):
    def __init__(self):
        super(PalmLocNet, self).__init__()
        self.plnet1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= 16, kernel_size= 5, stride =1, padding= 2),
       #     torch.nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =2)
        )
        self.plnet2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
        #    torch.nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.plnet3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
        #    torch.nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.plnet4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
        #    torch.nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.plnet5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
        #    torch.nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.outlinear = nn.Sequential(
            nn.Linear(256 * 15* 15, 6400),
            torch.nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.Linear(6400, 128),
            torch.nn.Dropout(0.5),  # drop 50% of the neuron
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.plnet1(x)
        x = self.plnet2(x)
        x = self.plnet3(x)
        x = self.plnet4(x)
        x = self.plnet5(x)
        x = x.view(x.size(0),-1)
        output = self.outlinear(x)
        return output

#自定义损失函数
class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
    def MyGIoU(self, pred_loc, truth_loc):
        if use_gpu:
            pred_loc = pred_loc.cuda()
            truth_loc = truth_loc.cuda()
        else:
            pass
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
        if use_gpu:
            zer = torch.zeros(x_g1.shape).cuda()
        else:
            zer = torch.zeros(x_g1.shape)
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
      #  locMSEloss = (pred_loc-truth_loc).pow(2).sum()/(4*truth_loc.shape[0])
        gIoUloss = 1- self.MyGIoU(pred_loc, truth_loc).sum()/truth_loc.shape[0]
       # myloss = locMSEloss+gIoUloss
        myloss = gIoUloss
        return myloss


#训练以及保存模型数据
def train_PalmLocNet(train_loader, test_x, test_y):
    if os.path.exists(args.MODELFOLDER + 'train_params_best.pth'):
        print('reload the last best model parameters')
        if use_gpu:
            palnet = PalmLocNet()
            palnet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth'))
            palnet = palnet.cuda()
        else:
            palnet = PalmLocNet()
            palnet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth',map_location='cpu'))
    else:
        print('It is the first time to train the model!')
        if use_gpu:
            palnet = PalmLocNet()
            palnet = palnet.cuda()
        else:
            palnet = PalmLocNet()

    optimizer = torch.optim.Adam(palnet.parameters(),lr= args.LR)
    loss_func = Myloss()

   # compare_loss = [0]

    for epoch in range(args.EPOCH):
        for step, (x, y) in enumerate(train_loader):
            if use_gpu:
                b_x = x.cuda()
                b_y = y.cuda()
            else:
                b_x = x
                b_y = y

            output = palnet(b_x)
            loss = loss_func(output, b_y)
            optimizer.zero_grad()  # 将上一步梯度值清零
            loss.backward()  # 求此刻各参数的梯度值
            optimizer.step()  # 更新参数

            # 可视化模型结构
            with SummaryWriter(log_dir='PLNet01') as w:
                w.add_graph(palnet,(b_x,))
                w.add_scalar('Train', loss, global_step=(epoch+1)*100+step)

            if step % 50 == 0:
                palnet.eval()
                test_output = palnet(test_x)
                test_loss_func = Myloss()
                #test_GIoU = test_loss_func.MyGIoU(test_output,test_y).sum() /(test_y.shape[0])
                test_GIoU = 1- test_loss_func(test_output,test_y)
                test_locMSEloss = (test_output - test_y).pow(2).sum() /(4*test_y.shape[0])
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
           # compare_loss.append(loss)
           # print('compare_loss.append:', compare_loss)
            if loss < best_loss:
                torch.save(palnet.state_dict(), args.MODELFOLDER + 'train_params_best.pth')
                print('save the best trained model in epoch', epoch)
                best_loss = loss
                print('new best_loss:', best_loss)
            else:
                print('no better in this epoch', epoch)

#只加载训练好的参数
def test_PalmLocNet(test_x, test_y):
    if use_gpu:
        PLNet = PalmLocNet()
        PLNet.eval()
        PLNet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth'))
        PLNet = PLNet.cuda()
    else:
        PLNet = PalmLocNet()
        PLNet.eval()
        PLNet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth',map_location='cpu'))
    test_output_Plnet = PLNet(test_x)
    test_loss_func_Plnet = Myloss()
    test_GIoU_Plnet = test_loss_func_Plnet.MyGIoU(test_output_Plnet, test_y).sum()/test_y.shape[0]
    test_locMSEloss_Plnet = (test_output_Plnet - test_y).pow(2).sum() / (4*test_y.shape[0])
    test_loss_Plnet = test_loss_func_Plnet(test_output_Plnet, test_y)
    print('test GIoU: %.4f' % test_GIoU_Plnet, '\n'
          'test locMSEloss: %.4f' % test_locMSEloss_Plnet,'\n'
          'total test loss: %.4f' % test_loss_Plnet)
    return test_output_Plnet

def testpic():
    oupt = test_PalmLocNet(test_x, test_y)
    fh = open(args.PICTUREFOLDER + 'testset/' + 'test.txt', 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        imgs.append([words[0], words[1], words[2], words[3], words[4]])

    k = 0
    for p in imgs:
        img = cv2.imread(p[0])
        # 画矩形框
        # 预测框
        cv2.rectangle(img, (oupt[k][0], oupt[k][1]), (oupt[k][2], oupt[k][3]), (0, 255, 0), 4)
        cv2.rectangle(img, (int(p[1]), int(p[2])), (int(p[3]), int(p[4])), (0, 0, 255), 4)
        k += 1
        cv2.imwrite(args.PICTUREFOLDER + 'testset/' + str(k) + '_test_truth.jpg', img)



def testvideolocnet():
    if use_gpu:
        PLNet = PalmLocNet()
        PLNet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth'))
        PLNet = PLNet.cuda()
    else:
        PLNet = PalmLocNet()
        PLNet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth',map_location='cpu'))
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            print(frame.shape)
            frame = cv2.resize(frame, (480, 480))
             #用permute改变高维tensor的维数位置
            tframe = torch.from_numpy(frame).permute(2,0,1)
             #加一维
            tframe = tframe.unsqueeze(0)
            tframe = tframe.float()
            outloc = PLNet(tframe)
            print(outloc)
            cv2.rectangle(frame, (1, 60), (100, 200), (0, 255, 0), 4)
            cv2.rectangle(frame, (outloc[0][0], outloc[0][1]), (outloc[0][2], outloc[0][3]), (0, 255, 0), 4)
            frame = cv2.resize(frame, (640, 480))
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
        testpic()







