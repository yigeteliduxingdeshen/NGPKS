import csv
import torch as t
import numpy as np

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        md_data = []
        md_data += [[float(i) for i in row] for row in reader]
        return t.FloatTensor(md_data)




if __name__ == "__main__":
    M_FSM = read_csv('../data/m-m.csv')  # MiRNA功能相似性矩阵
    D_SSM = read_csv('../data/d-d.csv')  # 疾病语义相关矩阵
    MD_MDM = read_csv('../data/m-d.csv')  # MiRNA-disease关联矩阵
    gamall = 1
    gamadd = 1
    alpha = 0.1
    KD = np.zeros(shape=(D_SSM.shape[0], D_SSM.shape[1]))  # 初始化疾病矩阵
    KM = np.zeros(shape=(M_FSM.shape[0], M_FSM.shape[1]))  # 初始化miRNA矩阵
     # calculate gamal for Gaussian kernel calculation
    sd = np.zeros(shape=(D_SSM.shape[0]))  # 初始化疾病矩阵
    for i in range(D_SSM.shape[0]):
        sd[i] = np.linalg.norm(MD_MDM[:, i])** 2
    gamal = D_SSM.shape[0] / sum(sd.conj().T)*gamall

    # calculate gamal for Gaussian kernel calculation
    sm = np.zeros(shape=(M_FSM.shape[0]))  # 初始化miRNA矩阵
    for i in range(M_FSM.shape[0]):
        sm[i] = np.linalg.norm(MD_MDM[i,:])** 2
    gamad = M_FSM.shape[0] / sum(sm.conj().T) * gamadd
    # calculate Gaussian interaction profile kernel similarity for disease
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
            KD[i][j] = np.exp((1 - alpha)/2 *(-gamal * (np.linalg.norm(MD_MDM[:, i] - MD_MDM[:, j]))** 2) + alpha /2 *(-gamal * (np.linalg.norm(MD_MDM[:, i] - MD_MDM[:, j]))** 2))
    # calculate Gaussian interaction profile kernel similarity for miRNA
    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            KM[i][j] = np.exp((1 - alpha)/2 *(-gamad * (np.linalg.norm(MD_MDM[i,:] -  MD_MDM[j,:]))** 2) + alpha /2 * (-gamad *(np.linalg.norm(MD_MDM[i,:] -  MD_MDM[j,:]))** 2))
    np.savetxt('../data/D_GSM.txt',KD)
    np.savetxt('../data/M_GSM.txt',KM)
