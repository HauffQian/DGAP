import json
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import itertools
import pickle
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
# 读取JSON文件


def weights_init_(m):
    if isinstance(m,nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight,gain = 1)
        torch.nn.init.constant_(m.bias, 0)

class ThreeLayerFC(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ThreeLayerFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()

        self.fc1_1 = nn.Linear(state_dim + action_dim, 20480)
        self.fc1_2 = nn.Linear(20480, 20480)
        self.fc1_3 = nn.Linear(20480, 2048)
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, 2)
        self.apply(weights_init_)
        
    def forward(self, inputs):
        d1 = F.relu(self.fc1_1(inputs))
        # d2 = F.relu(self.fc1_2(inputs[: , -16:]))
        # d = torch.cat([d1, d2], 1)
        d = F.relu(self.fc1_2(d1))
        d = F.relu(self.fc1_3(d))
        d = F.relu(self.fc2(d))
        # d = F.sigmoid(self.fc3(d))
        d = F.tanh(self.fc3(d))
        # d = torch.clip(d, 0.1, 0.9)
        # d = torch.clip(d, 0.1, 0.9)
        return d





def eval():
    
    # 加载模型
    loaded_model = ThreeLayerFC(input_size, hidden_size)
    loaded_model.load_state_dict(torch.load('mlp_model.pth'))
    loaded_model.eval()  # 设置为评估模式

    # 使用模型进行预测
    # 假设有一个名为test_input的输入张量，大小为(batch_size, input_size)
    with torch.no_grad():
        outputs = loaded_model(test_input)

def print_dimensions(lst, depth=0):
    if isinstance(lst, list):
        print("Depth:", depth, "Length:", len(lst))
        for sub_list in lst:
            print_dimensions(sub_list, depth + 1)
    else:
        print("Depth:", depth, "Length:", 1)








def convert(document,tokenizer,score):
    max_input_length = 1024
    max_action_length = 16
    inputs = []
    labels = []
    with open(document, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            # 获取input和action
                input_text = data['input']
                action_text = data['action']
                
                # Tokenize input并补足长度到1024
                encoded_input = tokenizer(input_text, max_length=160, padding='max_length', truncation=True, return_tensors='pt')
                
                # Tokenize action并补足长度到16
                encoded_action = tokenizer(action_text, max_length=16, padding='max_length', truncation=True, return_tensors='pt')
                
                # 将编码后的输入拼接
                input_ids = torch.cat((encoded_input.input_ids.squeeze(), encoded_action.input_ids.squeeze()), dim=-1)
                
                # 将拼接后的输入添加到列表中
                inputs.append(input_ids)
                
                # 获取score作为标签
                labels.append(float(score))

            except json.JSONDecodeError as e:
                # 打印错误信息并继续处理下一行
                print("JSON解析错误:", e)
                continue  
    return inputs,labels



def convert_st(document,model_path,score):
    max_input_length = 1024
    max_action_length = 16
    inputs = []
    labels = []
    model = SentenceTransformer(model_path)
    model.to("cuda")
    print("Model Loaded")
    cnt = 0
    with open(document, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            # 获取input和action
                input_text = data['input']
                action_text = data['action']
                
                embeddings_1 = model.encode(input_text, normalize_embeddings=True)
                
                # Tokenize action并补足长度到16
                embeddings_2 = model.encode(action_text, normalize_embeddings=True)
                
                # 将编码后的输入拼接
                input_ids = np.concatenate((embeddings_1.squeeze(), embeddings_2.squeeze()), axis=0)
                
                # 将拼接后的输入添加到列表中
                inputs.append(input_ids)
                
                # 获取score作为标签
                labels.append(float(score))
                cnt += 1
                if (cnt % 100 == 0):
                    print("Loading........")
            except json.JSONDecodeError as e:
                # 打印错误信息并继续处理下一行
                print("JSON解析错误:", e)
                continue  
    return inputs,labels



def hook_fn(module, grad_input, grad_output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Gradient input shape: {grad_input[0]}")
    print(f"Gradient output shape: {grad_output[0]}")



def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("使用的设备:", device)
        
    tokenizer = AutoTokenizer.from_pretrained('/home/qha2sgh/SwiftSage/flant5')
    
#-----------------------------------------------------------------------------------
    
    # inputse, labelse= convert("outputexpert.json",tokenizer,1) 
    # inputso, labelso= convert("outputsub.json",tokenizer,0) 


    # inputse = torch.stack(inputse)
    # inputso = torch.stack(inputso)
    # labelse = torch.tensor(labelse)
    # labelso = torch.tensor(labelso)
    
    
    # print("Shape of inputse:", inputse.shape)
    # print("Shape of inputso:", inputso.shape)
    # print("Shape of labelse:", labelse.shape)
    # print("Shape of labelso:", labelso.shape)
    # # 创建数据集
    # datasete = torch.utils.data.TensorDataset(inputse,labelse)
    # dataseto = torch.utils.data.TensorDataset(inputso,labelso)


    # with open('datasete.pkl', 'wb') as f:
    #     pickle.dump(datasete, f)
    # with open('dataseto.pkl', 'wb') as f:
    #     pickle.dump(dataseto, f)
#----------------------------------------------------
    # with open('com_datasete.pkl', 'rb') as f:
    #     datasete = pickle.load(f)

    # # 加载 dataseto.pkl 文件
    # with open('com_dataseto.pkl', 'rb') as f:
    #     dataseto = pickle.load(f)



    train_ratio = 0.9  # 训练集占总数据集的比例
    test_ratio = 1 - train_ratio  # 测试集占总数据集的比例


    train_sizee = int(train_ratio * len(datasete))
    test_sizee = len(datasete) - train_sizee
    train_sizeo = int(train_ratio * len(dataseto))
    test_sizeo = len(dataseto) - train_sizeo
    
    
    # 分割数据集
    train_datasete, test_datasete = torch.utils.data.random_split(datasete, [train_sizee, test_sizee])
    train_dataseto, test_dataseto = torch.utils.data.random_split(dataseto, [train_sizeo, test_sizeo])

    batch_size = 64

    # 加载器
    train_loadere = DataLoader(train_datasete, batch_size=batch_size, shuffle=True)
    test_loadere = DataLoader(test_datasete, batch_size=batch_size, shuffle=True)
    train_loadero = DataLoader(train_dataseto, batch_size=batch_size, shuffle=True)
    test_loadero = DataLoader(test_dataseto, batch_size=batch_size, shuffle=True)
    loadero_iter = itertools.cycle(train_loadero)
    loadero_iter_test = itertools.cycle(test_loadero)
    
    
    learning_rate = 0.001  # 学习率
    num_epochs = 100  # 训练轮数
    input_size = 1040  # 例如，对于MNIST数据集，输入大小为28x28=784
    hidden_size = 512

    # 创建模型实例
    model = Discriminator(1024, 16)
    model.to(device)

    # 打印模型结构
    print(model)

    # criterion = nn.MSELoss()
    # 定义Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # hook = model.fc1_1.register_forward_hook(hook_fn)
    best_accuracy = 0 
    # 开始训练
    for epoch in range(num_epochs):
        total_loss = 0.0
        # 训练模型
        model.train()
        for (inputs1, labels1), (inputs2, labels2) in tqdm(zip(train_loadere, train_loadere), desc=f'Epoch {epoch + 1}/{num_epochs} (Training)', unit='batch'):
            # 在这里使用 inputs1, labels1 作为 train_loader 的批次数据
            # 使用 inputs2, labels2 作为 train_loadero 的批次数据
            # 前向传播
            inputs1_tensor = inputs1.squeeze()
            inputs2_tensor = inputs2.squeeze()

            inputs1, inputs2 = inputs1_tensor.float().to(device), inputs2_tensor.float().to(device)
            labels1 = labels1.to(device)  # 假设标签是张量
            labels2 = labels2.to(device)  # 假设标签是张量
            d_e = model(inputs1)
            d_o = model(inputs2)
            # print("d_e: ",d_e,'\n',"d_o: ",d_o)
            # if d_o.size(0) != d_e.size(0):
            #     end_index = min(d_e.size(0),d_o.size(0))
            #     d_o = d_o[:end_index]
            #     d_e = d_e[:end_index]
            # d_loss_e = -torch.log(d_e)
            # d_loss_o = -torch.log(1 - d_o) / 0.5 + torch.log(1 - d_e)
            # d_loss = torch.mean(d_loss_e + d_loss_o)

            d_e = d_e.squeeze()
            d_o = d_o.squeeze()
            loss_e = F.mse_loss(d_e, labels1)
            loss_o = F.mse_loss(d_o, labels2)
            d_loss = torch.mean(loss_e) + torch.mean(loss_o)
            optimizer.zero_grad()
            d_loss.backward()
            optimizer.step()
            # 反向传播和优化

            
            total_loss += d_loss
            for name, parms in model.named_parameters():
                print('-->name:', name, '-->grad_requires:', parms.requires_grad, '--weight:', torch.mean(parms.data).item(), end=' ')
                if parms.grad is not None:
                    print('-->grad_value:', torch.mean(parms.grad).item())
                else:
                    print('-->grad_value: None')


        # 计算训练集上的平均损失
        train_loss = total_loss / len(train_loadere)+len(train_loadero)
        
        # 在测试集上评估模型性能
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (inputs1, labels1), (inputs2, labels2) in tqdm(zip(test_loadere, loadero_iter_test), desc=f'Epoch {epoch + 1}/{num_epochs} (Testing)', unit='batch'):
                inputs1_tensor = inputs1.squeeze()
                inputs2_tensor = inputs2.squeeze()
                inputse, inputso = inputs1_tensor.float().to(device), inputs2_tensor.float().to(device)
                outputse, outputso = model(inputse), model(inputso)
                total += outputso.size(0)
                total += outputse.size(0)
                correct_e = (outputse > 0.8).sum().item()
                correct_o = (outputso < 0.2).sum().item()
                total_correct = correct_e + correct_o

        # 计算测试集上的准确率
        test_accuracy = total_correct / total
        if test_accuracy > best_accuracy:

            torch.save(model.state_dict(), 'best_mlp_model_411.pth')
            best_accuracy = test_accuracy  # 更新最佳准确率
            
        # 打印每轮训练的平均损失和测试集准确率
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Test Accuracy: {correct} / {total}")



    torch.save(model.state_dict(), 'mlp_model_411.pth')

def train_with_medium():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("使用的设备:", device)
        
    tokenizer = AutoTokenizer.from_pretrained('/home/qha2sgh/SwiftSage/flant5')
    
#-----------------------------------------------------------------------------------
    
    # inputse, labelse= convert("outputexpert_13_comp.json",tokenizer,1) 
    # inputso, labelso= convert("outputsub_comp.json",tokenizer,0) 


    # inputse = torch.stack(inputse)
    # inputso = torch.stack(inputso)
    # labelse = torch.tensor(labelse)
    # labelso = torch.tensor(labelso)
    
    
    # print("Shape of inputse:", inputse.shape)
    # print("Shape of inputso:", inputso.shape)
    # print("Shape of labelse:", labelse.shape)
    # print("Shape of labelso:", labelso.shape)
    # # 创建数据集
    # datasete = torch.utils.data.TensorDataset(inputse,labelse)
    # dataseto = torch.utils.data.TensorDataset(inputso,labelso)


    # with open('com_datasete_13.pkl', 'wb') as f:
    #     pickle.dump(datasete, f)
    # with open('com_dataseto.pkl', 'wb') as f:
    #     pickle.dump(dataseto, f)
#----------------------------------------------------
    with open('com_datasete_13.pkl', 'rb') as f:
        datasete = pickle.load(f)

    # # 加载 dataseto.pkl 文件
    with open('com_dataseto.pkl', 'rb') as f:
        dataseto = pickle.load(f)

    # 将加载的数据转换为 TensorDataset 对象

    train_ratio = 0.9  # 训练集占总数据集的比例
    test_ratio = 1 - train_ratio  # 测试集占总数据集的比例

    # 计算数据集中每个部分的数量
    train_sizee = int(train_ratio * len(datasete))
    test_sizee = len(datasete) - train_sizee
    train_sizeo = int(train_ratio * len(dataseto))
    test_sizeo = len(dataseto) - train_sizeo
    
    
    # 使用random_split分割数据集
    train_datasete, test_datasete = torch.utils.data.random_split(datasete, [train_sizee, test_sizee])
    train_dataseto, test_dataseto = torch.utils.data.random_split(dataseto, [train_sizeo, test_sizeo])

    batch_size = 24

    # 创建数据加载器
    train_loadere = DataLoader(train_datasete, batch_size=batch_size, shuffle=True)
    test_loadere = DataLoader(test_datasete, batch_size=batch_size, shuffle=True)
    train_loadero = DataLoader(train_dataseto, batch_size=batch_size, shuffle=True)
    test_loadero = DataLoader(test_dataseto, batch_size=batch_size, shuffle=True)
    loadero_iter = itertools.cycle(train_loadero)
    loadero_iter_test = itertools.cycle(test_loadero)

    
    learning_rate = 0.0001  # 学习率
    num_epochs = 100  # 训练轮数
    input_size = 1040  
    hidden_size = 512

    # 创建模型实例
    model = Discriminator(160, 16)
    model.to(device)

    # 打印模型结构
    print(model)

    # criterion = nn.MSELoss()
    # 定义Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.8)
    # hook = model.fc1_1.register_forward_hook(hook_fn)
    best_accuracy = 0 
    
    # 开始训练
    for epoch in range(num_epochs):
        flag = True
        total_loss = 0.0
        cnt = 0
        # 训练模型
        model.train()
        for (inputs1, labels1), (inputs2, labels2) in tqdm(zip(train_loadere, loadero_iter), desc=f'Epoch {epoch + 1}/{num_epochs} (Training)', unit='batch'):
            # 使用 inputs1, labels1 作为 train_loader 的批次数据
            # 使用 inputs2, labels2 作为 train_loadero 的批次数据

            inputs1_tensor = inputs1.squeeze()
            inputs2_tensor = inputs2.squeeze()

            inputs1, inputs2 = inputs1_tensor.float().to(device), inputs2_tensor.float().to(device)
            labels1 = labels1.long().to(device)  
            labels2 = labels2.long().to(device) 
            d_e = model(inputs1)
            d_o = model(inputs2)
            d_e = d_e.squeeze()
            d_o = d_o.squeeze()
            # print("d_e: ",d_e,'\n',"d_o: ",d_o)
            # if d_o.size(0) != d_e.size(0):
            #     end_index = min(d_e.size(0),d_o.size(0))
            #     d_o = d_o[:end_index]
            #     d_e = d_e[:end_index]
            # d_loss_e = -torch.log(d_e)
            # d_loss_o = -torch.log(1 - d_o) / 0.5 + torch.log(1 - d_e)
            # d_loss = torch.mean(d_loss_e + d_loss_o)



            d = torch.cat((d_e, d_o),0)
            la = torch.cat((labels1, labels2),0)
            loss_e = F.cross_entropy(d,la)
            # if cnt < 5:
            #     print("Loss:---------------------",loss_e,"-------------------------")
            #     print("Total Loss:---------------------",total_loss,"-------------------------")
            #     cnt+=1
            optimizer.zero_grad()
            loss_e.backward()
            optimizer.step()
            
            if (d_o[:, 0] > 0).any():
                matching_indices = torch.nonzero(d_o[:, 0] > 0)
                for index in matching_indices:
                    row_index = index.item()

                    print("iutputse: ",inputs1[row_index,:10])
                    print("iutputso: ",inputs2[row_index,:10])
                    print("outputse: ",d_e[row_index,:])
                    print("outputso: ",d_o[row_index,:])
                # print("label1: " , labels1[-1] , "label2: " , labels2[-1])
            
            total_loss += loss_e
            if flag:
                for name, parms in model.named_parameters():
                    print('-->name:', name, '-->grad_requires:', parms.requires_grad, '--weight:', torch.mean(parms.data).item(), end=' ')
                    if parms.grad is not None:
                        print('-->grad_value:', torch.mean(parms.grad).item())
                    else:
                        print('-->grad_value: None')
                print("outputse: ",d_e[-3:,:])
                print("outputso: ",d_o[-3:,:])
                flag = False


        train_loss = total_loss / (len(train_loadere)+len(train_loadero))
        # print("Total Loss: ",total_loss,"simples: ",len(train_loadere)+len(train_loadero),"train_loss: ",train_loss)
        # 在测试集上评估模型性能
        model.eval()
        total = 0

        with torch.no_grad():
            cnt = 0
            total_correcte = 0
            total_correcto = 0
            for (inputs1, labels1), (inputs2, labels2) in tqdm(zip(test_loadere, loadero_iter_test), desc=f'Epoch {epoch + 1}/{num_epochs} (Testing)', unit='batch'):
                if inputs1.size(0) != inputs2.size(0):
                    continue
                inputs1_tensor = inputs1.squeeze()
                inputs2_tensor = inputs2.squeeze()
                inputse, inputso = inputs1_tensor.float().to(device), inputs2_tensor.float().to(device)
                outputse, outputso = model(inputse), model(inputso)
                # if cnt < 5:
                #     print("test outputse: ",d_e[-3:,:])
                #     print("test outputso: ",d_o[-3:,:])
                #     cnt += 1
                total += outputso.size(0)
                total += outputse.size(0)
                correct_e = (outputse[:,1] > 0.8).sum().item()
                correct_o = (outputso[:,0] > 0.8).sum().item()
                total_correcte += correct_e
                total_correcto += correct_o
            flag = False

        # 计算测试集上的准确率
        test_accuracy = (total_correcte+total_correcto) / total
        if test_accuracy > best_accuracy:

            torch.save(model.state_dict(), 'best_mlp_model_416.pth')
            best_accuracy = test_accuracy  # 更新最佳准确率
            
        # 打印每轮训练的平均损失和测试集准确率
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Test Accuracy E: {total_correcte} / {total}, Test Accuracy O: {total_correcto} / {total}")
        # scheduler.step()


    torch.save(model.state_dict(), 'mlp_model_416.pth')

def statis():
    
    with open('datasete_13.pkl', 'rb') as f:
        datasete = pickle.load(f)

    # # 加载 dataseto.pkl 文件
    with open('dataseto.pkl', 'rb') as f:
        dataseto = pickle.load(f)

    # 将加载的数据转换为 TensorDataset 对象

    train_ratio = 0.9  # 训练集占总数据集的比例
    test_ratio = 1 - train_ratio  # 测试集占总数据集的比例

    # 计算数据集中每个部分的数量
    train_sizee = int(train_ratio * len(datasete))
    test_sizee = len(datasete) - train_sizee
    train_sizeo = int(train_ratio * len(dataseto))
    test_sizeo = len(dataseto) - train_sizeo
    
    
    # 使用random_split分割数据集
    train_datasete, test_datasete = torch.utils.data.random_split(datasete, [train_sizee, test_sizee])
    train_dataseto, test_dataseto = torch.utils.data.random_split(dataseto, [train_sizeo, test_sizeo])

    batch_size = 4

    # 创建数据加载器
    train_loadere = DataLoader(train_datasete, batch_size=batch_size, shuffle=True)
    test_loadere = DataLoader(test_datasete, batch_size=batch_size, shuffle=True)
    train_loadero = DataLoader(train_dataseto, batch_size=batch_size, shuffle=True)
    test_loadero = DataLoader(test_dataseto, batch_size=batch_size, shuffle=True)
    loadero_iter = itertools.cycle(train_loadero)
    loadero_iter_test = itertools.cycle(test_loadero)
    
    
    learning_rate = 0.0001  # 学习率
    num_epochs = 1  # 训练轮数
    input_size = 1040  
    hidden_size = 512
    for (inputs1, labels1), (inputs2, labels2) in tqdm(zip(train_loadere, loadero_iter), desc=f'Epoch 1 / 1 (Eval)', unit='batch'):
        for dim in range(4):
            different_indices = torch.nonzero(inputs1[dim, :] != inputs2[dim, :]).squeeze()
            if len(different_indices) > 0:
                print(f"在维度{dim}下不同的值:")
                for index in different_indices:
                    print(f"    inputs1[{dim}, {index}]={inputs1[dim, index]}, inputs2[{dim}, {index}]={inputs2[dim, index]}")

def PCAS():
    inputse = []
    inputso = []
    all_inputs = []

    with open('com_datasete_13.pkl', 'rb') as f:
        datasete = pickle.load(f)

    # # 加载 dataseto.pkl 文件
    with open('com_dataseto.pkl', 'rb') as f:
        dataseto = pickle.load(f)
    
    
    datae_loader = DataLoader(datasete, batch_size=32)
    for inputs, labels in datae_loader:
        all_inputs.append(inputs)
        inputse.append(inputs)
    inputse = torch.cat(inputse, dim=0)

    datao_loader = DataLoader(dataseto, batch_size=32)
    for inputs, labels in datao_loader:
        all_inputs.append(inputs)
        inputso.append(inputs)
    inputso = torch.cat(inputso, dim=0)
    data = torch.cat(all_inputs, dim=0).numpy()
    

    pca = PCA(n_components=2)  # 设置要降到的维度数
    transformed_data = pca.fit_transform(data)

    # 绘制PCA降维后的数据
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=['b'] * len(inputse) + ['r'] * len(inputso), alpha=0.5)
    plt.title("PCA Transformed Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()
    
def PCAS_stfo():
    inputse = []
    inputso = []
    all_inputs = []
    model_path = "/home/qha2sgh/SwiftSage/feature_embedding/bge-large-en-v1.5"
    inputse, labelse= convert_st("outputexpert_13_comp.json",model_path,1) 
    inputso, labelso= convert_st("outputsub_comp.json",model_path,0) 

    np.savez('result_st.npz', inputse=inputse)
    np.savez('result_st.npz', inputso=inputso)    

    
    # print("Shape of inputse:", inputse.shape)
    # print("Shape of inputso:", inputso.shape)
    # print("Shape of labelse:", labelse.shape)
    # print("Shape of labelso:", labelso.shape)
    # 创建数据集

    # loaded_data = np.load('result_st.npz')
    # print(loaded_data.files)
    # inputse = loaded_data['inputse']
    # inputso = loaded_data['inputso']
    
 
    data = np.concatenate((inputse,inputso),axis = 0)

    
    

    

    pca = PCA(n_components=2)  # 设置要降到的维度数
    transformed_data = pca.fit_transform(data)

    # 绘制PCA降维后的数据
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], c=['b'] * len(inputse) + ['r'] * len(inputso), alpha=0.5)
    plt.title("PCA Transformed Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


def main():
    # train()
    # train_with_medium()
    # statis()
    # PCAS()
    PCAS_stfo()

if __name__ =='__main__':
    main()