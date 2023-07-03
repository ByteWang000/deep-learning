import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from model import Generator1, Discriminator1




def main():
    # MNIST数据集格式：单通道28x28大小的手写数字图片
    image_size = [1,28,28]
    latent_dim = 100 # 潜在向量的大小（噪声）
    batch_size = 8

    dataset = torchvision.datasets.MNIST("mnist_data", train=True, download=True,
                                         transform = transforms.Compose(
                                        [
                                                transforms.Resize(28),
                                                transforms.ToTensor(),
                                         ]  
                                         )
                                         )
    # 加载数据集
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # 创建网络模型
    generator = Generator1(latent_dim= latent_dim,image_size=image_size)
    discriminator = Discriminator1(image_size=image_size)
    # 定义优化器，损失函数
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999),weight_decay=0.0001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999),weight_decay=0.0001)

    loss_fn = nn.BCELoss()
    labels_one = torch.ones(batch_size,1)
    labels_zero = torch.zeros(batch_size,1)

    #训练网络
    epoch = 100
    for epoch in range(epoch):
        for i, minibatch in enumerate(dataloader):
            # [batch_size, 1, 28, 28]
            # 1: channel, 28: height, 28: width
            gt_images, _ = minibatch

            z = torch.randn(batch_size, latent_dim)

            fake_images = generator(z)

            g_loss = loss_fn(discriminator(fake_images), labels_one)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            real_loss = loss_fn(discriminator(gt_images), labels_one)
            fake_loss = loss_fn(discriminator(fake_images.detach()), labels_zero)
            d_loss = (real_loss + fake_loss)

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            if i % 50 == 0:
                print(f"step:{len(dataloader) * epoch + i}, g_loss:{g_loss.item()}, d_loss:{d_loss.item()}, real_loss:{real_loss.item()}, fake_loss:{fake_loss.item()}")
            if i % 500 == 0:
                image = fake_images[:16].data
                torchvision.utils.save_image(image, f"image_{len(dataloader)*epoch+i}.png", nrow=4)
    print('Finished Training !')
    torch.save(generator.state_dict(),'./generator.pth')
    torch.save(discriminator.state_dict(),'./discriminator.pth')

if __name__ == '__main__':
    main()