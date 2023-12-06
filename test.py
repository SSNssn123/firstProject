import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from mydataset import makeMyDataSet, myDateSet

root="../data/dataset"

myTestData2Numpy, myTestLabels = makeMyDataSet(root, "test")
myDateSetTest = myDateSet(myTestData2Numpy, myTestLabels, transforms.ToTensor())
test_loader = DataLoader(dataset=myDateSetTest, batch_size=64, shuffle=False)


# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5, 1, 2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(64*4*4, 64),
#             nn.Linear(64, 10)
#         )

#     def forward(self, x):
#         x = self.model(x)
#         return x
    

#加载模型 
model = torch.load("ssn_29_gpu.pth", map_location=torch.device('cpu'))
print(model)
myDateSetTest = torch.reshape(myDateSetTest, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(myDateSetTest)
print(output)

print(output.argmax(1))
