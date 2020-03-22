#coding=utf8
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from sklearn import metrics
from datetime import timedelta
from os.path import join as os_join
import pandas as pd


from utils import tokenizer, build_vocab
from utils import build_dataset
from PublicConfig import PublicConfig


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

class TextCNNConfig:
    def __init__(self, public_config):
        self.batch_size = 64
        self.filter_num = 256
        self.filter_size = [2, 3, 4]
        self.embedding_size = 300
        self.drop_out = 0.5
        self.learning_rate = 1e-4
        self.epoch = 5
        self.save_path = os_join(public_config.current_path, "models", "textcnn.ckpt")
        self.require_improvement = 1000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding = torch.tensor(np.load(public_config.out_embedding_path)["embeddings"].astype("float32"))


public_config = PublicConfig()
textcnn_config = TextCNNConfig(public_config)

class TextCNN(nn.Module):
    def __init__(self, public_config, textcnn_config):
        super(TextCNN, self).__init__()
        self.class_number = public_config.class_number
        self.embedding = torch.nn.Embedding.from_pretrained(textcnn_config.embedding, freeze=False)
        self.convs = nn.ModuleList(modules=[
            nn.Conv2d(1, textcnn_config.filter_num, (k, textcnn_config.embedding_size)) \
                                    for k in textcnn_config.filter_size
            ])
        
        self.dropout = nn.Dropout(textcnn_config.drop_out)
        self.fc = nn.Linear(len(textcnn_config.filter_size) * textcnn_config.filter_num, self.class_number)

    def forward(self, x):
        embed = self.embedding(x)       # [batch, seq_len, embed_size]
        embed = embed.unsqueeze(1)      # [batch, 1, seq_len, embed_size]
        tmp_lst = []
        for conv in self.convs:
            conv_out = conv(embed)                  # [batch, filter_number, seq_len - k + 1, 1]
            conv_out = F.relu(conv_out).squeeze(3)  # [batch, filter_number, seq_len -k + 1]
            conv_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)  # [batch, filter_number]
            tmp_lst.append(conv_out)
        out = torch.cat(tmp_lst, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def train(model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=textcnn_config.learning_rate)
    total_batch = 0
    dev_best_loss = float("inf")
    flag = False
    last_improve = 0
    for i in range(textcnn_config.epoch):
        print(f"epoch {i}")
        for j, (X, y) in enumerate(train_iter):
            outputs = model(X)
            loss = F.cross_entropy(outputs, y)

            optim.zero_grad()
            loss.backward()
            optim.step()
            # print("loss:", loss.item())

            if total_batch % 100 == 0:
                true = y.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, dev_iter)
                model.train()
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), textcnn_config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
            total_batch += 1
            # print("total batch: ", total_batch)
            if total_batch - last_improve > textcnn_config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(model, test_iter)

def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=public_config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def test(model, test_iter):
    # test
    model.load_state_dict(torch.load(textcnn_config.save_path))
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)

def predict(model, predict_iter):
    model.load_state_dict(torch.load(textcnn_config.save_path))
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in predict_iter:
            outputs = model(texts)
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    predict_class = list(map(lambda x: public_config.class_list[x], predict_all))
    df = pd.DataFrame({"y":labels_all, "y_hat": predict_all, "predict_class": predict_class})
    df.to_csv("test_predict.csv", index=False)

def main():
    train_features, train_labels = build_dataset(public_config, public_config.train_path)
    dev_features, dev_labels = build_dataset(public_config, public_config.dev_path)
    test_features, test_labels = build_dataset(public_config, public_config.test_path)

    predict_features, predict_labels = build_dataset(public_config, "NewsData/predict.txt")

    train_set = TensorDataset(*(torch.tensor(train_features, device=textcnn_config.device), torch.tensor(train_labels, device=textcnn_config.device)))
    dev_set = TensorDataset(*(torch.tensor(dev_features, device=textcnn_config.device), torch.tensor(dev_labels, device=textcnn_config.device)))
    test_set = TensorDataset(*(torch.tensor(test_features, device=textcnn_config.device), torch.tensor(test_labels, device=textcnn_config.device)))
    predict_set = TensorDataset(*(torch.tensor(predict_features, device=textcnn_config.device), torch.tensor(predict_labels, device=textcnn_config.device)))
    

    train_iter = DataLoader(train_set, batch_size=textcnn_config.batch_size, shuffle=True)
    dev_iter = DataLoader(dev_set, batch_size=textcnn_config.batch_size, shuffle=True)
    test_iter = DataLoader(test_set, batch_size=textcnn_config.batch_size, shuffle=False)
    predict_iter = DataLoader(predict_set, batch_size=1, shuffle=False)

    model = TextCNN(public_config, textcnn_config).to(textcnn_config.device)
    # train(model, train_iter, dev_iter, test_iter)
    test(model, test_iter)
    predict(model, predict_iter)


if __name__ == "__main__":
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    main()



    