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
#from d2l import *
from torchvision.io import read_image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_exp_name = 'resnet50'
data_train = pd.read_csv('train.csv')
data_train.label.value_counts()
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

def train_FOLD(net, train_iter,valid_iter,trainer,epochs,patience):
    '''patience 为多次参数不更新时停止的步数'''
#     def init_weights(m):
#         if type(m) == nn.Linear or type(m) == nn.Conv2d:
#             nn.init.xavier_uniform_(m.weight)
#     net.apply(init_weights)
    print('training on',device)
    net.to(device)
    #optimizer = torch.optim.SGD(net.parameters(),lr = learning_rate,weight_decay=weight_decay)
#     optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate,weight_decay=weight_decay)
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
            trainer.zero_grad()
            X,y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat,y)
            l.backward()
            trainer.step()
            
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
            print(f'Best model fodun at epoch {epoch},save model')
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

# 预训练模型
resnet = torchvision.models.resnet50(pretrained=True)
#改变最后全连接层结构
finetune = torchvision.models.resnet50(pretrained=True)
#finetune = torchvision.models.resnext101_32x8d(pretrained=True)
finetune.fc = nn.Linear(finetune.fc.in_features, 176)
nn.init.xavier_uniform_(finetune.fc.weight)
# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

train_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    normalize])

test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ToTensor(),
    normalize])

#  对数据集进行拆分，划分为训练集和验证集（8:2）
L = int(len(data_train)/10 * 8)
train_csv = data_train.iloc[:L].reset_index()
valid_csv = data_train.iloc[L:].reset_index()
print(f'train data size:{train_csv.shape},test data size:{valid_csv.shape}')
batch_size = 256
train_iter, valid_iter = create_dls(train_csv, valid_csv, train_augs, train_augs, batch_size)

def train_fine_tuning(net,learning_rate,epochs,param_group):
    loss = nn.CrossEntropyLoss(reduction='none')
    if param_group:
        params_lx = [param for name,param in net.named_parameters()
                    if name not in ['fc.weight','fc.bias']]
        trainer = torch.optim.AdamW([{'params':params_lx},
                                  {'params':net.fc.parameters(),'lr':learning_rate * 10}],
                                 lr = learning_rate, weight_decay=1e-4)
    else:
        trainer = torch.optim.AdamW(net.parameters(),lr = learning_rate,weight_decay=1e-4)
    patience = 200
#     for i in range(5):
        
    train_iter, valid_iter = create_dls(train_csv, valid_csv, train_augs, train_augs, batch_size)
    train_FOLD(net,train_iter,valid_iter,trainer,epochs,patience)

learning_rate = 5e-4
epochs = 50
train_fine_tuning(finetune,learning_rate,epochs,True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = pd.read_csv('test.csv')
test_ds = LeavesDataset(test_data,test_augs)
test_iter = DataLoader(test_ds,batch_size = 256,shuffle = False)

model_best = finetune.to(device)
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
data.to_csv(f'submission_{_exp_name}.csv')