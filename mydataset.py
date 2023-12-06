import torchvision
import numpy as np
import random
from torch.utils.data import Dataset

def makeMyDataSet(root, type):

    train_data = torchvision.datasets.CIFAR10(root=root
                                            ,train=True
                                            ,transform=torchvision.transforms.ToTensor()
                                            #  ,download=True
                                            )
    test_data = torchvision.datasets.CIFAR10(root=root
                                            ,train=False 
                                            ,transform=torchvision.transforms.ToTensor()
                                            #  ,download=True
                                            )
    

    trainData2Numpy = np.array(train_data.data)
    trainDataLabels = train_data.targets
    
    testData2Numpy = np.array(test_data.data)
    testDataLabels = test_data.targets

    allData2Numpy = np.concatenate((trainData2Numpy, testData2Numpy), axis=0)
    allDataLabels = trainDataLabels + testDataLabels

    index = [i for i in range(len(allDataLabels))]
    random.shuffle(index)

    if type == "train":
        myTrainData2Numpy = allData2Numpy[:int(len(allDataLabels)/10*9)]
        myTrainLabels = allDataLabels[:int(len(allDataLabels)/10*9)]
        return myTrainData2Numpy, myTrainLabels
    # elif type == "val":
    #     myValData2Numpy = allData2Numpy[int(len(allDataLabels)/10*8):int(len(allDataLabels)/10*9)]
    #     myValLabels = allDataLabels[int(len(allDataLabels)/10*8):int(len(allDataLabels)/10*9)]
    #     return myValData2Numpy, myValLabels
    elif type == "test":
        myTestData2Numpy = allData2Numpy[int(len(allDataLabels)/10*9):]
        myTestLabels = allDataLabels[int(len(allDataLabels)/10*9):]
        return myTestData2Numpy, myTestLabels

   

class myDateSet(Dataset):
    def __init__(self, myData2Numpy, myLabel, transform=None):
        super(myDateSet, self).__init__()
        self.myData2Numpy = myData2Numpy
        self.myLabel = myLabel
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.myData2Numpy[index]), self.myLabel[index]
    
    def __len__(self):
        return self.myData2Numpy.shape[0]