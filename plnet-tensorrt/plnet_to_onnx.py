import torch
from model.PLNet import vggPalmLocNet
from torchvision import models
import onnx

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dummy_input = torch.randn((1, 3, 224, 224), device='cuda')

vgg = models.vgg16_bn(pretrained=False)
PLNet = vggPalmLocNet(vgg)
PLNet.eval()
modelpath = '/home/aaronwd/桌面/engineering/plnet-tensorrt/model/train_params_best.pth'
if torch.cuda.is_available():
    PLNet.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath).items()})
else:
    PLNet.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(modelpath,map_location='cpu').items()})

PLNet.to(device)

print(PLNet)

input_names = ['input']
output_names = ['output']
torch.onnx.export(PLNet, dummy_input, 'PLNet.onnx', verbose=True, input_names=input_names, output_names=output_names)

#######################check #####################
net = onnx.load('PLNet.onnx')
onnx.checker.check_model(net)
graph = onnx.helper.printable_graph(net.graph)
print('---'*20)
print(graph)

############## param #####################
params = list(PLNet.parameters())

k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))


