from models.PLNet import vggPalmLocNet
from data.mydataset import MyDataset
from lossfuc.myloss import Myloss

############################
import torch
import torch.nn as nn
import argparse
import os
from torchvision import transforms, models
from torch.utils.data import DataLoader

######pic_size = 480
#######pic_resize =224

#设置超参数
parser = argparse.ArgumentParser(description='super params')
parser.add_argument('-e','--EPOCH', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('-b','--BATCH_SIZE', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('-l','--LR', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('-m','--MODELFOLDER',type= str, default='./checkpoints/',
                help="folder to store model")
parser.add_argument('-p','--PICTUREFOLDER',type= str, default='./picture_total/',
                help="folder to store trained picture")

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if os.path.exists(args.PICTUREFOLDER+'trainset/'+'train.txt') and os.path.exists(args.PICTUREFOLDER+'testset/'+'test.txt') :
    print('train.txt and test.txt have been existed')
    transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])

    train_data = MyDataset(txt=args.PICTUREFOLDER + 'trainset/' + 'train.txt', transform=transforms)
    test_data = MyDataset(txt=args.PICTUREFOLDER + 'testset/' + 'test.txt', transform=transforms)
    train_loader = DataLoader(dataset=train_data, batch_size=args.BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=10, num_workers=8)
else:
    print('you need to prepare your train.txt and test.txt first!')

#测试数据
for k, (tx, ty) in enumerate(test_loader):
    test_x = tx.to(device)
    test_y = ty.to(device)/480

# 训练以及保存模型数据
def train_PalmLocNet(train_loader, test_x, test_y):
    if os.path.exists(args.MODELFOLDER + 'train_params_best.pth'):
        print('reload the last best model parameters')

        vgg = models.vgg16_bn(pretrained=False)
        pal = vggPalmLocNet(vgg)

        if torch.cuda.is_available():
            pal.load_state_dict(
                {k.replace('module.', ''): v for k, v in
                 torch.load(args.MODELFOLDER + 'train_params_best.pth').items()})
        else:
            pal.load_state_dict(
                {k.replace('module.', ''): v for k, v in
                 torch.load(args.MODELFOLDER + 'train_params_best.pth', map_location='cpu').items()})
        # print(vgg)
        # print(pal)

        if torch.cuda.device_count() > 1:
            print('lets use', torch.cuda.device_count(), 'GPUs')
            palnet = nn.DataParallel(pal)
        else:
            palnet = pal
        palnet.to(device)

    else:
        print('It is the first time to train the model!')
        vgg = models.vgg16_bn(pretrained=False)

        if torch.cuda.is_available():
            vgg.load_state_dict(
                {k.replace('module.', ''): v for k, v in
                 torch.load(args.MODELFOLDER + 'vgg16_bn-6c64b313.pth').items()})
        else:
            vgg.load_state_dict(
                {k.replace('module.', ''): v for k, v in
                 torch.load(args.MODELFOLDER + 'vgg16_bn-6c64b313.pth', map_location='cpu').items()})

        pal = vggPalmLocNet(vgg)

        if torch.cuda.device_count() > 1:
            print('lets use', torch.cuda.device_count(), 'GPUs')
            palnet = nn.DataParallel(pal)
        else:
            palnet = pal
        palnet.to(device)

    optimizer = torch.optim.Adam(palnet.parameters(), lr=args.LR)
    loss_func = Myloss()

    for epoch in range(args.EPOCH):
        for step, (x, y) in enumerate(train_loader):
            b_x = x.to(device)
            b_y = y.to(device) / 480
            output = palnet(b_x)

            loss = loss_func(output, b_y)
            optimizer.zero_grad()  # 将上一步梯度值清零
            loss.backward()  # 求此刻各参数的梯度值
            optimizer.step()  # 更新参数

            if step % 100 == 0:
                palnet.eval()
                test_output = palnet(test_x)
                print('test_output[0]', test_output[0])
                print('test_y[0]', test_y[0])
                test_loss_func = Myloss()
                test_GIoU = test_loss_func.MyGIoU(test_output, test_y).sum() / (test_y.shape[0])
                test_locMSEloss = ((test_output - test_y).pow(2).sum() / (4 * test_y.shape[0]))
                test_loss = test_loss_func(test_output, test_y)
                print('Epoch', epoch, '\n'
                                      'train loss: %.4f' % loss.data.cpu().numpy(), '\n'
                                                                                    'test GIoU: %.4f' % test_GIoU, '\n'
                                                                                                                   'test locMSEloss: %.4f' % test_locMSEloss,
                      '\n'
                      'total test loss: %.4f' % test_loss)
                palnet.train()

        # 检查是否有模型文件夹，没有就自行创建一个
        if not os.path.isdir(args.MODELFOLDER):
            os.makedirs(args.MODELFOLDER)

        # 保存训练loss最小的模型参数（按周期记步）
        if epoch == 0:
            if os.path.exists(args.MODELFOLDER + 'train_params_best.pth'):
                print('exist the train_params_best.pth!')
                pass
            else:
                print('first make the train_params_best.pth')
                torch.save(palnet.state_dict(), args.MODELFOLDER + 'train_params_best.pth')
            best_loss = loss
            print('best_loss in epoch 0:', best_loss)
        else:
            if loss < best_loss:
                torch.save(palnet.state_dict(), args.MODELFOLDER + 'train_params_best.pth')
                print('save the best trained model in epoch', epoch)
                best_loss = loss
                print('new best_loss:', best_loss)
            else:
                print('no better in this epoch', epoch)

# 运行训练模型
if __name__ == '__main__':
    train_PalmLocNet(train_loader, test_x, test_y)

