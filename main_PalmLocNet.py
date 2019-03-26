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

######pic_size = 64

#设置超参数
parser = argparse.ArgumentParser(description='super params')
parser.add_argument('-e','--EPOCH', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('-b','--BATCH_SIZE', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-l','--LR', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('-m','--MODELFOLDER',type= str, default='./model/',
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
args = parser.parse_args()

#检测是否有利用的gpu环境
use_gpu = torch.cuda.is_available()
print('use GPU:',use_gpu)

#数据预处理
pass


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
            imgs.append((words[0], words[1]))
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

train_data = MyDataset(txt=args.PICTUREFOLDER+'train.txt', transform=transforms.ToTensor())
test_data = MyDataset(txt=args.PICTUREFOLDER+'test.txt', transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_data, batch_size=args.BATCH_SIZE, shuffle=True)
#test_loader = DataLoader(dataset=test_data, batch_size=args.BATCH_SIZE)

# test_x =
# test_y =
#
# for (tx, ty) in enumerate(test_loader):
#     if use_gpu:
#         test_x = tx.cuda()
#         test_y = ty.cuda()
#     else:
#         test_x = tx
#         test_y = ty
测试数据
if use_gpu:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:1].cuda()/255.
    test_y = test_data.test_labels[:1].cuda()
else:
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:1]/255.
    test_y = test_data.test_labels[:1]

#神经网络建模
class PalmLocNet(nn.Module):
    def __init__(self):
        super(PalmLocNet, self).__init__()
        self.plnet1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= 16, kernel_size= 5, stride =1, padding= 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =2)
        )
        self.plnet2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.outlinear = nn.Sequential(
            nn.Linear(32 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.plnet1(x)
        x = self.plnet2(x)
        x = x.view(x.size(0),-1)
        output = self.outlinear()
        return output

#自定义损失函数
class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
    def MyGIoU(self, pred_loc, truth_loc):
        x_p1 = pred_loc[0]
        y_p1 = pred_loc[1]
        x_p2 = pred_loc[2]
        y_p2 = pred_loc[3]
        #print(x_p1, y_p1, x_p2, y_p2)

        x_g1 = truth_loc[0]
        y_g1 = truth_loc[1]
        x_g2 = truth_loc[2]
        y_g2 = truth_loc[3]
        #print(x_g1, y_g1, x_g2, y_g2)

        A_g = (x_g2 - x_g1) * (y_g2 - y_g1)
        A_p = (x_p2 - x_p1) * (y_p2 - y_p1)
        #print(A_g, A_p)

        x_I1 = max(x_p1, x_g1)
        x_I2 = min(x_p2, x_g2)
        y_I1 = max(y_p1, y_g1)
        y_I2 = min(y_p2, y_g2)
        #print(x_I1, x_I2, y_I1, y_I2)

        if x_I2 > x_I1 and y_I2 > y_I1:
            I = (x_I2 - x_I1) * (y_I2 - y_I1)
        else:
            I = torch.tensor([0])
        #print(I)

        x_C1 = min(x_p1, x_g1)
        x_C2 = max(x_p2, x_g2)
        y_C1 = min(y_p1, y_g1)
        y_C2 = max(y_p2, y_g2)
        #print(x_C1, x_C2, y_C1, y_C2)

        A_c = (x_C2 - x_C1) * (y_C2 - y_C1)
        U = A_g + A_p - I
        myIoU = I / U
        myGIoU = myIoU - (A_c - U) / A_c
        #print('IoU: %.4f, gIoU: %.4f' % (myIoU, myGIoU))
        return myGIoU

    def forward(self, pred_loc, truth_loc):
        locMSEloss = (pred_loc-truth_loc).pow(2).sum()/4
        gIoUloss = 1- self.MyGIoU(pred_loc, truth_loc)
        myloss = locMSEloss+gIoUloss
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
            palnet.load_state_dict(torch.load(args.MODELFOLDER + 'train_params_best.pth'))
    else:
        if use_gpu:
            palnet = PalmLocNet()
            palnet = palnet.cuda()
        else:
            palnet = PalmLocNet()

    optimizer = torch.optim.Adam(palnet.parameters(),lr= args.LR)
    loss_func = Myloss()

    compare_loss = []
    test_compare_loss = []

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
                test_output = palnet(test_x)
                test_loss_func = Myloss()
                test_GIoU = test_loss_func.MyGIoU(test_output,test_y)
                test_locMSEloss = (test_output - test_y).pow(2).sum() / 4
                test_loss = test_loss_func(test_output,test_y)
                print('Epoch', epoch, '\n'
                      'train loss: %.4f' % loss.data.cpu().numpy(),'\n'
                      'test GIoU: %.4f' % test_GIoU,'\n'
                      'test locMSEloss: %.4f' % test_locMSEloss,'\n'
                      'total test loss: %.4f' % test_loss)

        # 检查是否有模型文件夹，没有就自行创建一个
        if not os.path.isdir(args.MODELFOLDER):
            os.makedirs(args.MODELFOLDER)

        #保存训练loss最小的模型参数（按周期记步）
        if epoch == 0:
            torch.save(palnet.state_dict(), args.MODELFOLDER + 'train_params_best.pth')
            compare_loss[0] = loss
        else:
            # append方法并没有返回值
            compare_loss.append(loss)
            if compare_loss[epoch]<compare_loss[epoch-1]:
                torch.save(palnet.state_dict(), args.MODELFOLDER + 'train_params_best.pth')
                print('save the best trained model in epoch', epoch)
            else:
                print('no better in this epoch', epoch)

        # 保存测试loss最小的模型参数（按周期记步）
        if epoch == 0:
            torch.save(palnet.state_dict(), args.MODELFOLDER + 'test_params_best.pth')
            if use_gpu:
                PLNet0 = PalmLocNet()
                PLNet0.load_state_dict(torch.load(args.MODELFOLDER + 'test_params_best.pth'))
                PLNet0 = PLNet0.cuda()
            else:
                PLNet0 = PalmLocNet()
                PLNet0.load_state_dict(torch.load(args.MODELFOLDER + 'test_params_best.pth'))

            test_output_Plnet0 = PLNet0(test_x)
            test_loss_func_Plnet0 = Myloss()
            test_loss_Plnet0 = test_loss_func_Plnet0(test_output_Plnet0, test_y)
            test_compare_loss[0] = test_loss_Plnet0
        else:
            torch.save(palnet.state_dict(), args.MODELFOLDER + 'test_params_epoch.pth')
            if use_gpu:
                PLNet0 = PalmLocNet()
                PLNet0.load_state_dict(torch.load(args.MODELFOLDER + 'test_params_epoch.pth'))
                PLNet0 = PLNet0.cuda()
            else:
                PLNet0 = PalmLocNet()
                PLNet0.load_state_dict(torch.load(args.MODELFOLDER + 'test_params_epoch.pth'))

            test_output_Plnet0 = PLNet0(test_x)
            test_loss_func_Plnet0 = Myloss()
            test_loss_Plnet0 = test_loss_func_Plnet0(test_output_Plnet0, test_y)
            test_compare_loss.append(test_loss_Plnet0)

            if test_compare_loss[epoch]<test_compare_loss[epoch-1]:
                torch.save(palnet.state_dict(), args.MODELFOLDER + 'test_params_best.pth')
                print('save the best test model in epoch', epoch)
            else:
                print('no better test in this epoch', epoch)

#只加载训练好的参数
def test_PalmLocNet(test_x, test_y):
    if use_gpu:
        PLNet = PalmLocNet()
        PLNet.load_state_dict(torch.load(args.MODELFOLDER + 'test_params_best.pth'))
        PLNet = PLNet.cuda()
    else:
        PLNet = PalmLocNet()
        PLNet.load_state_dict(torch.load(args.MODELFOLDER + 'test_params_best.pth'))
    test_output_Plnet = PLNet(test_x)
    test_loss_func_Plnet = Myloss()
    test_GIoU_Plnet = test_loss_func_Plnet.MyGIoU(test_output_Plnet, test_y)
    test_locMSEloss_Plnet = (test_output_Plnet - test_y).pow(2).sum() / 4
    test_loss_Plnet = test_loss_func_Plnet(test_output_Plnet, test_y)
    print('test GIoU: %.4f' % test_GIoU_Plnet, '\n'
          'test locMSEloss: %.4f' % test_locMSEloss_Plnet,'\n'
          'total test loss: %.4f' % test_loss_Plnet)

#运行训练以及测试模型
if (__name__ == '__main__') and args.TrainOrNot:
    train_PalmLocNet(train_loader, test_x, test_y)

if (__name__ == '__main__') and (not args.TrainOrNot):
    test_PalmLocNet(test_x, test_y)








