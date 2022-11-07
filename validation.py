import torch as t
import torch.utils.data as Data
import numpy as np
from torch import nn
from sklearn.metrics import roc_curve, roc_auc_score,precision_score,f1_score,recall_score, precision_recall_curve
from sklearn.metrics import confusion_matrix
import math
from sklearn.metrics import auc
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
device = t.device("cuda" if t.cuda.is_available() else "cpu")
val_feature = np.loadtxt('./data/test_feature.txt')
val_label = np.loadtxt('./data/test_label.txt')
val_feature = t.FloatTensor(val_feature)
val_label = t.FloatTensor(val_label)
model = FullyConnectedNuralNetwork().to(device)
# map_location:指定设备，cpu或者GPU
model.load_state_dict(t.load("./save_model_rs_dataset/NGPKSMDA/train_model_10.pth", map_location="cpu"))
val_data = Data.TensorDataset(val_feature, val_label)
val_loader = Data.DataLoader(dataset=val_data, batch_size=500, shuffle=True, num_workers=0)
loss_fn = nn.MSELoss()
def compute_accuracy_and_loss(model, data_loader, device):
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device)
        output = model(features).reshape(-1)
        y_pred = np.around(output.detach().cpu().numpy(), 0).astype(int)
        # 处理输出数据
        currnet_loss = loss_fn(output, targets)
        CM = confusion_matrix(targets.cpu().numpy(), y_pred)
        FPR, TPR, _ = roc_curve(targets.cpu().numpy(), output.detach().cpu().numpy(), pos_label=1)
        AUC = roc_auc_score(targets.cpu().numpy(), output.detach().cpu().numpy())
        pre = precision_score(targets.cpu().numpy(), y_pred, average='macro')
        recall = recall_score(targets.cpu().numpy(), y_pred, average='macro')
        precision_1, recall_1, threshold = precision_recall_curve(targets.cpu().numpy(), y_pred)
        AUPR = auc(recall_1,precision_1)
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
    return AUC, currnet_loss.item(), pre, recall, f1, FPR, TPR, Acc, Sen, Spec, MCC, AUPR
# 计算测试精度
val_auc, val_loss, val_pre, val_recall, val_f1, val_FPR, val_TPR, val_Acc, val_Sen, val_Spec, val_MCC, val_AUPR = compute_accuracy_and_loss(model, val_loader, device=device)
print(f'Validation Loss.: {val_loss:.2f}')
print(f'Validation Auc.: {val_auc:.4f}')
print(f'Validation AUPR.: {val_AUPR:.4f}')
print(f'Validation pre.: {val_pre:.2f}')
print(f'Validation recall.: {val_recall:.2f}')
print(f'Validation F1.: {val_f1:.2f}')
print(f'Validation acc.: {val_Acc:.2f}')
print(f'Validation sen.: {val_Sen:.2f}')
print(f'Validation spec.: {val_Spec:.2f}')
print(f'Validation mcc.: {val_MCC:.2f}')
# np.savetxt('./data/linear2_10FCV_FPR.txt',val_FPR)
# np.savetxt('./data/linear2_10FCV_TPR.txt',val_TPR)