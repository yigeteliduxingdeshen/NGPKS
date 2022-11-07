import csv
import torch as t
import random
from matplotlib import pyplot as plt
import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from torch import nn
from sklearn.metrics import roc_curve, roc_auc_score,precision_score,f1_score,recall_score
from sklearn.metrics import confusion_matrix
import math
def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)
#获取边索引 返回长向量
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] >= 0.5:         #大于0.5相关联性大
                edge_index[0].append(i)
                edge_index[1].append(j)
    return t.LongTensor(edge_index)
#定义网络

class FullyConnectedNuralNetwork(nn.Module):
    def __init__(self):
        super(FullyConnectedNuralNetwork,self).__init__()
        self.hidden1=nn.Sequential(
                nn.Linear(in_features=878,out_features=256,bias=True),
                nn.ReLU())
        self.hidden2=nn.Sequential(
                nn.Linear(in_features=256,out_features=128,bias=True),
                nn.ReLU())
        self.hidden3=nn.Sequential(
                nn.Linear(in_features=128,out_features=1,bias=True),
                nn.Sigmoid())
    def forward(self,x):
        fc1=self.hidden1(x)
        fc2=self.hidden2(fc1)
        score=self.hidden3(fc2)
        return score


def compute_accuracy_and_loss(model, data_loader, device):
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        output = model(features).reshape(-1)
        y_pred =np.around(output.cpu().numpy(),0).astype(int)
        # 处理输出数据
        currnet_loss = loss_fn(output, targets)
        CM = confusion_matrix(targets.cpu().numpy(), y_pred)
        FPR, TPR, _ = roc_curve(targets.cpu().numpy(), output.cpu().numpy(), pos_label=1)
        AUC = roc_auc_score(targets.cpu().numpy(), output.cpu().numpy())
        pre = precision_score(targets.cpu().numpy(), y_pred, average='macro')
        recall = recall_score(targets.cpu().numpy(), y_pred, average='macro')
        f1 = f1_score(targets.cpu().numpy(), y_pred, average='macro')
        CM = CM.tolist()
        TN = CM[0][0]
        FP = CM[0][1]
        FN = CM[1][0]
        TP = CM[1][1]
        Acc = (TN + TP) / (TN + TP + FN + FP)
        Sen = TP / (TP + FN)
        Spec = TN / (TN + FP)
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    return AUC, currnet_loss.item(), pre, recall, f1, FPR, TPR, Acc, Sen, Spec, MCC

if __name__ == "__main__":
    # 加载数据集
    D_SSM = read_csv( './data/d-d.csv')      #疾病语义相关矩阵
    D_GSM = np.loadtxt('./data/D_GSM.txt')     #疾病高斯相互作用分布核相似度
    M_FSM = read_csv('./data/m-m.csv')      #MiRNA功能相似性矩阵
    M_GSM = np.loadtxt('./data/M_GSM.txt')     #MiRNA高斯相互作用分布核相似度
    MD_MDM = read_csv('./data/m-d.csv')  # MiRNA-disease关联矩阵
     #加入miRNA和disease高斯核相似度矩阵
    disease_feature = np.zeros(shape=(D_SSM.shape[0], D_SSM.shape[1]))  # 初始化疾病矩阵
    mirna_feature = np.zeros(shape=(M_FSM.shape[0], M_FSM.shape[1]))    # 初始化miRNA矩阵
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
            if D_SSM[i][j] == 0:  # 将相似度为0的部分替换为高斯相互作用轮廓核
                disease_feature[i][j] = D_GSM[i][j]
            else:
                disease_feature[i][j] = D_SSM[i][j]
    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if M_FSM[i][j] == 0:  # 将相似度为0的部分替换为高斯相互作用轮廓核
                mirna_feature[i][j] = M_GSM[i][j]
            else:
                mirna_feature[i][j] = M_FSM[i][j]
    #划分数据集
    #分别将关联度为0的和关联度为1的下标取出
    zero_index = []         #负样本集
    one_index = []          #正样本集
    for i in range(MD_MDM.size(0)):
        for j in range(MD_MDM.size(1)):
            if MD_MDM[i][j] < 1:
                zero_index.append([i, j, 0])
            if MD_MDM[i][j] >= 1:
                one_index.append([i, j, 1])
    zero_index = random.sample(zero_index,len(one_index))
    #将正样本相应的特征找出
    positive_sample=[]
    for i in range(len(one_index)):
        positive_sample.append(list(np.concatenate((mirna_feature[one_index[i][0]],disease_feature[one_index[i][1]]),axis=0)))
    # positive_sample = np.array(positive_sample)
    #将负样本相应的特征找出
    negative_sample=[]
    for j in range(len(zero_index)):
        negative_sample.append(list(np.concatenate((mirna_feature[zero_index[j][0]],disease_feature[zero_index[j][1]]),axis=0)))
    sample = np.array(positive_sample + negative_sample)
    #标签
    label = []
    for i in range(len(sample)):
        if i < 5430:
            label.append(1)
        else:
            label.append(0)
    label = np.array(label)
    # 如果GPU可用，利用GPU进行训练
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    # 实例化网络
    model = FullyConnectedNuralNetwork().to(device)
    #定义训练参数
    # 4. 损失函数
    loss_fn = nn.MSELoss()
    # 学习率
    learning_rate = 0.001
    # 5. 优化器
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    # 学习率衰减⽅法：学习率每隔 step_size 个 epoch 变为原来的 gamma
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.01)
    # 训练轮数
    epoch = 50

    # 保存训练过程中的loss和精度
    train_auc_lst, test_auc_lst = [], []
    train_loss_lst, tset_loss_lst = [], []
    train_FPR_lst, train_TPR_lst = [], []
    test_FPR_lst, test_TPR_lst = [], []
    # 记录训练过程中最大的精度
    max_train_auc = 0
    max_test_auc = 0

    final_train_auc = 0
    final_test_auc = 0
    k = 0
    kf = KFold(n_splits=10,shuffle=True,random_state=42)
    for train_index,test_index in kf.split(sample):
        print("---------开始第{}折交叉验证---------".format((k + 1)))
        train_data=Data.TensorDataset(t.FloatTensor(sample[train_index]),t.FloatTensor(label[train_index]))
        train_loader=Data.DataLoader(dataset=train_data,batch_size=500,shuffle=True,num_workers=1)
        test_data=Data.TensorDataset(t.FloatTensor(sample[test_index]),t.FloatTensor(label[test_index]))
        test_loader=Data.DataLoader(dataset=test_data,batch_size=500,shuffle=True,num_workers=1)
        start_time = time.time()
        #开始训练
        for i in range(epoch):
            print("---------开始第{}轮训练，本轮学习率为：{}---------".format((i + 1), lr_scheduler.get_last_lr()))
            # 记录每轮训练批次数，每100次进行一次输出
            count_train = 0
            # 训练步骤开始
            model.train()  # 将网络设置为训练模式，当网络包含 Dropout, BatchNorm时必须设置，其他时候无所谓
            for (features, targets) in train_loader:
                # 将特征和标签移动到指定设备上
                features = features.to(device)
                targets = targets.to(device)

                # 梯度清零，也就是把loss关于weight的导数变成0.
                # 进⾏下⼀次batch梯度计算的时候，前⼀个batch的梯度计算结果，没有保留的必要了。所以在下⼀次梯度更新的时候，先使⽤optimizer.zero_grad把梯度信息设置为0。
                optimizer.zero_grad()

                # 获取网络输出
                output = model(features).reshape(-1)
                # 获取损失
                loss = loss_fn(output, targets)
                # 反向传播
                loss.backward()
                # 训练
                optimizer.step()
                # 纪录训练次数
                count_train += 1
                # item()函数会直接输出值，比如tensor(5),会输出5
                if count_train % 100 == 0:
                    # 记录时间
                    end_time = time.time()
                    print(f"训练批次{count_train}/{len(train_loader)}，loss：{loss.item():.3f}，用时：{(end_time - start_time):.2f}")
            # 将网络设置为测试模式，当网络包含 Dropout, BatchNorm时必须设置，其他时候无所谓
            model.eval()
            with t.no_grad():
                # 计算训练精度
                train_auc, train_loss , train_pre, train_recall, train_f1, train_FPR, train_TPR, train_Acc, train_Sen, train_Spec, train_MCC = compute_accuracy_and_loss(model, train_loader,device=device)
                # 更新最高精度
                if train_auc > max_train_auc:
                    max_train_auc = train_auc

                # 计算测试精度
                test_auc, test_loss, test_pre, test_recall, test_f1, test_FPR, test_TPR, test_Acc, test_Sen, test_Spec, test_MCC = compute_accuracy_and_loss(model, test_loader, device=device)
                # 更新最高精度
                if test_auc > max_test_auc:
                    max_test_auc = test_auc

                # 收集训练过程精度和loss
                train_loss_lst.append(train_loss)
                train_auc_lst.append(train_auc)
                tset_loss_lst.append(test_loss)
                test_auc_lst.append(test_auc)

                train_FPR_lst.append(train_FPR.tolist())
                train_TPR_lst.append(train_TPR.tolist())
                test_FPR_lst.append(test_FPR.tolist())
                test_TPR_lst.append(test_TPR.tolist())

                print(f'Epoch: {i + 1:03d}/{epoch:03d}')
                print(f'Train Loss.: {train_loss:.2f}' f' | Validation Loss.: {test_loss:.2f}')
                print(f'Train Auc.: {train_auc:.2f}' f' | Validation Auc.: {test_auc:.2f}')
                print(f'Train pre.: {train_pre:.2f}' f' | Validation pre.: {test_pre:.2f}')
                print(f'Train recall.: {train_recall:.2f}' f' | Validation recall.: {test_recall:.2f}')
                print(f'Train F1.: {train_f1:.2f}' f' | Validation F1.: {test_f1:.2f}')
                print(f'Train acc.: {train_Acc:.2f}' f' | Validation acc.: {test_Acc:.2f}')
                print(f'Train sen.: {train_Sen:.2f}' f' | Validation sen.: {test_Sen:.2f}')
                print(f'Train spec.: {train_Spec:.2f}' f' | Validation spec.: {test_Spec:.2f}')
                print(f'Train mcc.: {train_MCC:.2f}' f' | Validation mcc.: {test_MCC:.2f}')
            if i == (epoch-1):
                final_train_auc = train_auc
                final_test_auc = test_auc
            # 训练计时
            elapsed = (time.time() - start_time) / 60
            print(f'本轮训练累计用时: {elapsed:.2f} min')

            # # 保存达标的训练的模型
            # if test_auc > 0.95:
            #     t.save(model.state_dict(), "./save_model_rs_dataset/linear4_10FCV/train_model_{}.pth".format(i))
            #     print("第{}次训练模型已保存".format(i + 1))

            # 更新学习率
            lr_scheduler.step()
        k += 1
    #绘制ROC曲线
    train_FPR_lst = np.array(train_FPR_lst[-1])
    train_TPR_lst = np.array(train_TPR_lst[-1])
    test_FPR_lst = np.array(test_FPR_lst[-1])
    test_TPR_lst = np.array(test_TPR_lst[-1])
    plt.figure()
    plt.plot(train_FPR_lst, train_TPR_lst, color='red', label='Train ROC curve (area =%0.2f)' % final_train_auc)
    plt.plot(test_FPR_lst, test_TPR_lst, color='blue', label='Validation ROC curve (area =%0.2f)' % final_test_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('10 Flod CV')
    plt.legend(loc="lower right")
    plt.show()
