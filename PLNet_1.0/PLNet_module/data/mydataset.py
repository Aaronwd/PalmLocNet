from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#测试时才用
import torch

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

if __name__ == "__main__":
    train_data = MyDataset('/home/aaron/桌面/PalmLocNet/picture/trainset/train.txt')