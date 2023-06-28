import torch
import torchvision.transforms as transforms
from PIL import Image  # python读取图像
from model import LeNet

transform = transforms.Compose(
    [transforms.Resize(32, 32),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
net = LeNet()
net.load_state_dict(torch.load('LeNet.pth'))
img = Image.open('1.jpg')
img = transforms(img)  # [c,h,w]
img = torch.unsqueeze(img, dim=0)  # [n,c,h,w]
with torch.no_grad():
    outputs = net(img)
    predict = torch.max(outputs, dim=1)[1].data.numpy()
    # predict = torch.softmax(outputs,dim=1)
print(classes[int(predict)])
