import pandas as pd
import numpy as np
import torch 
from torch import nn
import torchvision
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.io import read_image
_exp_name = "Alexnet"
data_train = pd.read_csv('train.csv')
print(data_train.label.value_counts())

label_list = sorted(list(set(data_train.label)))
label_map = {}
for i,label in enumerate(label_list):
    label_map[label] = i

#### 自定义数据集
class LeavesDataset(Dataset):
    def __init__(self,data, transform = None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data['image'])
    
    def __getitem__(self,idx):
        img = Image.open(self.data['image'][idx])#读取图片文件,[3x224x244]
        
#         par_path = os.path.join(os.getcwd(),self.data['image'][idx])
#         img = read_image(par_path)#将图片读取为张量形式，可以不用ToTensor()了

#         img = read_image(self.data['image'][idx])
        try:
            label = label_map[self.data['label'][idx]]#对应的标签
        except:
            label = -1
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label)


def create_dls(train_csv, test_csv, train_transform, test_transform, batch_size):
    train_ds = LeavesDataset(train_csv,train_transform)
    test_ds = LeavesDataset(test_csv,test_transform)
    train_dl = DataLoader(train_ds,batch_size = batch_size,shuffle = True)
    test_dl = DataLoader(test_ds,batch_size = batch_size,shuffle = True)
    return train_dl, test_dl

#  对数据集进行拆分，划分为训练集和验证集（8:2）
L = int(len(data_train)/10 * 8)
train_csv = data_train.iloc[:L].reset_index()
valid_csv = data_train.iloc[L:].reset_index()
print(f'train data size:{train_csv.shape},test data size:{valid_csv.shape}')
train_transform = T.Compose([T.Resize((224,224)),T.RandomHorizontalFlip(0.5),T.ColorJitter(hue=0.3),T.ToTensor()])
valid_transform = T.Compose([T.Resize((224,224)),T.ToTensor()])
batch_size = 256
train_iter, valid_iter = create_dls(train_csv,valid_csv,train_transform,valid_transform,batch_size)

class AlexNet(nn.Module):
    '''输入图片规模为3 x 224 x 224'''
    def __init__(self):
        super(AlexNet,self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(3,96,kernel_size=11,stride=4,padding=1),
                                  nn.BatchNorm2d(96),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,3),stride=2),
                                  nn.Conv2d(96,256,kernel_size=5,padding=2),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(), 
                                  nn.MaxPool2d((3,3),stride=2),
                                  nn.Conv2d(256,384,kernel_size=3,padding=1),
                                  nn.BatchNorm2d(384),
                                  nn.ReLU(),
                                  nn.Conv2d(384,384,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(384),
                                  nn.ReLU(),
                                  nn.Conv2d(384,256,kernel_size=3,padding=1),
                                   nn.BatchNorm2d(256),
                                  nn.ReLU(),
                                  nn.MaxPool2d((3,3),stride=2),
                                  nn.Flatten(),
                                  nn.Linear(6400,4096),nn.BatchNorm1d(4096),
                                   nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Linear(4096,4096),nn.ReLU(),
                                  nn.Dropout(0.5),
                                  nn.Linear(4096,176))
    def forward(self,X):
        out = self.layer(X)
        out = out.view(X.size()[0],-1)
        return out
alexnet = AlexNet()
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
alexnet.apply(init_weights)

def train_FOLD(net, train_iter,valid_iter,learning_rate,epochs,weight_decay,patience):
    '''patience 为多次参数不更新时停止的步数'''
#     def init_weights(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             nn.init.xavier_uniform_(m.weight)
#     net.apply(init_weights)
    print('training on',device)
    net.to(device)
    #optimizer = torch.optim.SGD(net.parameters(),lr = learning_rate,weight_decay=weight_decay)
    optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    num_batches = len(train_iter)
    # Initialize trackers, these are not parameters and should not be changed
    best_acc = 0
    stale = 0
    for epoch in range(epochs):
        train_los = []
        train_accs = []
        
        net.train()
        for (X,y) in train_iter:
            optimizer.zero_grad()
            X,y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            optimizer.step()
            
            #计算当前batch的准确度
            acc = (y_hat.argmax(dim=-1)==y).float().mean()
            
            #记录损失和准确度
            train_los.append(l.item())
            train_accs.append(acc)
        #计算一个epoch的平均损失和准确度
        train_loss = sum(train_los) / num_batches
        train_acc = sum(train_accs) / num_batches
        
        # ----------------------  Validation-----------------
        net.eval()
        
        valid_los = []
        valid_accs = []
        for X_val, y_val in valid_iter:
            X_val,y_val = X_val.to(device), y_val.to(device)
            with torch.no_grad():
                y_hat = net(X_val)
            l = loss(y_hat,y_val)
            
            acc = (y_hat.argmax(1)==y_val).float().mean()
            valid_accs.append(acc)
            valid_los.append(l.item())
        valid_loss = sum(valid_los) / len(valid_iter)
        valid_acc = sum(valid_accs) / len(valid_iter)
        print(f'epoch{epoch},train loss{train_loss:.3f},train acc:{train_acc:.3f},\
        valid loss{valid_loss:.3f}valid acc:{valid_acc:.3f}')
        
        ## save models, acc越高越好
        if valid_acc > best_acc:
            print(f'Best model found at epoch {epoch},save model')
            torch.save(net.state_dict(),f'{_exp_name}_best.ckpt')
            bets_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f'No improvement {patience} consecutive epochs, early stopping')
                break
        
    '''
    plt.figure(figsize = (5,4),dpi = 100)
    plt.plot(range(epochs),loss_item,'r--',label='loss')
    plt.plot(range(epochs),train_item,'b-.',label='train_acc')
    plt.plot(range(epochs),test_item ,'y',label='test_acc')
    plt.legend()
    #plt.savefig('test.png')
    '''
    #return loss_item, train_item, test_item

learning_rate = 1e-3
epochs = 100
weight_decay = 1e-5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
patience = 300

train_FOLD(alexnet,train_iter,valid_iter,learning_rate,epochs,weight_decay,patience)

test_data = pd.read_csv('test.csv')
test_ds = LeavesDataset(test_data,valid_transform)
test_iter = DataLoader(test_ds,batch_size = 64,shuffle = False)

model_best = AlexNet().to(device)
model_best.load_state_dict(torch.load(f'{_exp_name}_best.ckpt'))
model_best.eval()
predict = []
with torch.no_grad():
    for data,_ in test_iter:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(),axis=1)
        predict += test_label.squeeze().tolist()
        
data = pd.concat([test_data,pd.Series(predict)],axis=1)
data.columns = ['image','label']
new_dict = {v:k for k,v in label_map.items()}
data['label'] = data['label'].map(new_dict)
data.to_csv('submission_Alexnet.csv')