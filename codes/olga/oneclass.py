import torch
import numpy as np
from sklearn.metrics import classification_report

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_emb = None
        self.best_radius = None
        self.best_center = None
        self.best_epoch = None
        self.lowest_loss = None
        self.early_stop = False

    def step(self, score, cur_loss, epoch, radius, center, embs):
        if (self.best_score is None) or (self.lowest_loss is None):
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.best_emb = embs
            self.best_radius = radius
            self.best_center = center
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.best_emb = embs
            self.best_radius = radius
            self.best_center = center

            self.counter = 0

        return self.early_stop, self.best_emb, self.best_radius, self.best_center, self.best_epoch

def One_Class_GNN_prediction(center, radius, learned_representations, G, val_test, dic):

    with torch.no_grad():
        count = 0
        for node in G.nodes:
            G.nodes[node]['embedding_aocgnn'] = learned_representations[count].cpu().numpy()
            count+=1

        interest = []
        outlier = []
        for node in G.nodes:
            if G.nodes[node][val_test] == 1 and G.nodes[node]['label'] == 1:
                interest.append(G.nodes[node]['embedding_aocgnn'])
            elif G.nodes[node][val_test] == 1 and G.nodes[node]['label'] == 0:
                outlier.append(G.nodes[node]['embedding_aocgnn'])

        dist_int = np.sum((interest - center.cpu().numpy()) ** 2, axis=1)

        scores_int = dist_int - radius.cpu().numpy() ** 2

        dist_out = np.sum((outlier - center.cpu().numpy()) ** 2, axis=1)

        scores_out = dist_out - radius.cpu().numpy() ** 2

        preds_interest = [1 if score < 0 else -1 for score in scores_int]
        preds_outliers = [-1 if score > 0 else 1 for score in scores_out]

        y_true = [1] * len(preds_interest) + [-1] * len(preds_outliers)
        y_pred = list(preds_interest) + list(preds_outliers)
        if dic:
            return classification_report(y_true, y_pred, output_dict=dic)
        else:
            return print(classification_report(y_true, y_pred))

def one_class_loss(center, radius, learned_representations, mask):

    scores = anomaly_score(center, radius, learned_representations, mask)

    loss = torch.mean(torch.where(scores > 0, scores + 1, torch.exp(scores)))

    return loss

def anomaly_score(center, radius, learned_representations, mask):

    l_r_mask = torch.BoolTensor(mask)

    dist = torch.sum((learned_representations[l_r_mask] - center) ** 2, dim=1)

    scores = dist - radius ** 2

    return scores


def one_class_masking(G, train_val):
    train_mask = np.zeros(len(G.nodes), dtype='bool')
    unsup_mask = np.zeros(len(G.nodes), dtype='bool')

    normal_train_idx = []
    unsup_idx = []
    count = 0
    for node in G.nodes:
        if train_val:
            if G.nodes[node]['train'] == 1 or (G.nodes[node]['val'] == 1 and G.nodes[node]['label'] == 1):
                normal_train_idx.append(count)
            else:
                unsup_idx.append(count)
            count += 1
        else:
            if G.nodes[node]['train'] == 1:
                normal_train_idx.append(count)
            else:
                unsup_idx.append(count)
            count += 1

    train_mask[normal_train_idx] = 1
    unsup_mask[unsup_idx] = 1

    return train_mask, normal_train_idx, unsup_mask, unsup_idx
