##模型初始化
>model = Lenet()      # 初始化一个模型
<br>
## 方式一、保存与加载
> `# 保存模型,后缀可以为pt/pth`
> <br>
> torch.save(model.state_dict(), './model/model_state_dict.pth')
> <br>
> `# 加载模型`
> <br>
> model_test1 = Lenet()   # 加载模型时应先实例化模型
> <br>
> `# load_state_dict()函数接收一个字典，所以不能直接将'./model/model_state_dict.pth'传入，而是先使用load函数将保存的模型参数反序列化`
> <br>
> model_test1.load_state_dict(torch.load('./model/model_state_dict.pth'))
> <br>
> model_test1.eval()    # 模型推理时设置
> > <img src="https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/75e035d7956e4c8d843dd96428b2ab7a~tplv-k3u1fbpfcp-zoom-in-crop-mark:4536:0:0:0.awebp" >
## 方式二、（避免使用）
> `# 保存模型`
> <br>
> torch.save(model, './model/model.pt')    #这里我们保存模型的后缀名取.pt
> <br>
> `# 加载模型`
> model_test2 = torch.load('./model/model.pt')     
> model_test2.eval()   # 模型推理时设置
## 方式三、（可以中断后继续训练）
```python
# 保存checkpoint
torch.save({
            'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':loss
            
            }, './model/model_checkpoint.tar'    #这里的后缀名官方推荐使用.tar
            )
```
```python
# 加载checkpoint
model_checkpoint = Lenet()
optimizer =  optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
checkpoint = torch.load('./model/model_checkpoint.tar')    # 先反序列化模型
model_checkpoint.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```
```python
# 中断后重新开始继续训练
#4、创建网络模型
net = Net()
#5、设置损失函数、优化器
#损失函数
loss_fun = nn.CrossEntropyLoss()   #交叉熵
loss_fun = loss_fun.to(device)
#6、设置网络训练中的一些参数
total_train_step = 0   #记录总计训练次数
total_test_step = 0    #记录总计测试次数
Max_epoch = 10    #设计训练轮数

checkpoint = torch.load('./model/model_checkpoint_epoch_5.tar')    # 先反序列化模型
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
loss = checkpoint['loss']
#7、开始进行训练
for epoch in range(start_epoch+1, Max_epoch):
```
[详细内容查看](https://juejin.cn/post/7164004790416965640)
