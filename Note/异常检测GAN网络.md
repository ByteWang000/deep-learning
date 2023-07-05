# AnoGAN


# f-AnoGAN

# EGBAD
> 对比AnoGAN差别：
><br> 
>1、训练时还训练一个编码器，不在搜索潜在变量z，仅在推理时冻结编码器权重直接生成通过encoder潜在变量z。
> <br>
> 2、生成器生成假图像X‘和编码器生成的z’同时输入判别器。
> 以mnist数据集为例：
> <img src= https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/edaed3c57fd74fa48f95f197b8b649bf~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp alt="Image" style="width:70%">
><br>
> 编码器结构：
> <br>
> <img src=https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/40f94d168fd74f5abed13b3fad171205~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp alt="Image" style="width:70%">
> <br>
> 判别器：两个输入
><br>
> <img src=https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/bfdd68379ce94165b1b41adb2a7af462~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp alt="Image" style="width:70%">
###EGBAD缺陷计算得分
```python
## 定义缺陷计算的得分
def anomaly_score(input_image, fake_image, z_real, D):
    # Residual loss 计算:通过计算输入图像和生成的伪造图像之间的绝对差值的总和
    residual_loss = torch.sum(torch.abs(input_image - fake_image), (1, 2, 3))

    # Discrimination loss 计算:将输入图像和生成的伪造图像分别传递给判别器 D，获取特征表示。计算真实特征和伪造特征之间的绝对差值的总和.
    _, real_feature = D(input_image, z_real)
    _, fake_feature = D(fake_image, z_real)
    discrimination_loss = torch.sum(torch.abs(real_feature - fake_feature), (1))

    # 结合Residual loss和Discrimination loss计算每张图像的损失:将残差损失和判别损失结合起来，按照一定的权重进行加权求和
    total_loss_by_image = 0.9 * residual_loss + 0.1 * discrimination_loss

    return total_loss_by_image

```
# 