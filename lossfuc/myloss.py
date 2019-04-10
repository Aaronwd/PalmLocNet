import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


if __name__ == "__main__":
    x = torch.randn(2,4)
    y = torch.randn(2,4)
    myloss =Myloss()
    ls =myloss(x,y)