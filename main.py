import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchvision import transforms
from mydataset import makeMyDataSet, myDateSet
from mymodel import CNN

root="../data/dataset"

myTrainData2Numpy, myTrainLabels = makeMyDataSet(root, "train")
# myValData2Numpy, myValLabels = makeMyDataSet(root, "val")
# myTestData2Numpy, myTestLabels = makeMyDataSet(root, "test")
        
myDateSetTrain = myDateSet(myTrainData2Numpy, myTrainLabels, transforms.ToTensor())
# myDateSetVal = myDateSet(myValData2Numpy, myValLabels, transforms.ToTensor())
# myDateSetTest = myDateSet(myTestData2Numpy, myTestLabels, transforms.ToTensor())

train_loader = DataLoader(dataset=myDateSetTrain, batch_size=64, shuffle=True)
# val_loader = DataLoader(dataset=myDateSetVal, batch_size=64, shuffle=False)
# test_loader = DataLoader(dataset=myDateSetTest, batch_size=64, shuffle=False)

# 初始化模型
net = CNN()

# 定义交叉验证参数
k = 9
epochs = 2
batch_size = 64
lr = 1e-2

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr)

# 初始化k-fold
kf = KFold(n_splits=k, shuffle=True)

# 交叉验证训练
for fold, (train_index, val_index) in enumerate(kf.split(myDateSetTrain)):
    # print("TRAIN:", train_index, "TEST:", val_index)
    # print(len(train_index))
    # print(len(val_index))
    # print(fold)

    # 数据分为训练集和验证集
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_index)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_index)
    train_loader = DataLoader(myDateSetTrain, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(myDateSetTrain, batch_size=batch_size, sampler=val_sampler)
 
    # 训练模型
    for epoch in range(epochs):
        print("-------第 {} 轮训练开始-------".format(epoch+1))

        # 训练步骤开始
        net.train()
        total_train_step = 0
        total_train_loss = 0.0    # 每一batch的损失
        for i, data in enumerate(train_loader):
            inputs, targets = data
            outputs = net(inputs)
            loss = loss_fn(outputs, targets)
        
            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_train_step = total_train_step + 1
            if total_train_step % 100 == 0:
                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

        print('Epoch：[%d]/[%d] Loss:%.3f' % (epoch+1,epochs,total_train_loss)/(i+1))

        # 验证模型
        total_val_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data in val_loader:
                inputs1, targets1 = data
                outputs1 = net(inputs1)
                loss1 = loss_fn(outputs1, targets1)
                total_val_loss += loss1.item()
                accuracy = (outputs1.argmax(1) == targets1).sum()
                total_accuracy = total_accuracy + accuracy

        print("整体验证集上的Loss: {}".format(total_val_loss))
        print("整体验证集上的正确率: {}".format(total_accuracy/len(val_sampler)))


        torch.save(net, "ssn_{}.pth".format(epoch))
        print("模型已保存")




  

