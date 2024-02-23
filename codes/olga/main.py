import torch
import networkx as nx
from olga import OLGA
from oneclass import one_class_loss, one_class_masking, One_Class_GNN_prediction, EarlyStopping
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn.models.autoencoder import GAE
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import pandas as pd
import random
import sys
from pathlib import Path
import time
from utils import save_values, write_results, init_metrics
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

def train_olga(g, hidden1, hidden2, patience, lr, r, multi_task, epochs):
    loss_ocl = 0
    recon_loss_unsup = 0
    embeddings, losses_ocl, losses_rec, accuracies, losses = [], [], [], [], []
    best_embeddings, best_radius, best_center = [], 0, []
    c = [0] * hidden2
    center = torch.Tensor(c)
    radius = torch.Tensor([r])


    mask, t_mask, mask_unsup, t_mask_unsup = one_class_masking(g, False)
    G = from_networkx(g)
    g_unsup = g.subgraph(t_mask_unsup)
    G_unsup = from_networkx(g_unsup)
    model_ocl = OLGA(len(G.features[0]), [hidden1, hidden2])
    model = GAE(model_ocl)
    stopper = EarlyStopping(patience)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs+1):
        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        learned_representations = model.encode(G.features.float(), G.edge_index)

        if epoch < multi_task:
            loss = model.recon_loss(learned_representations, G.edge_index)
        else:
            loss_ocl = one_class_loss(center, radius, learned_representations, mask)

            recon_loss_unsup = model.recon_loss(learned_representations[mask_unsup], G_unsup.edge_index)

            loss = loss_ocl + recon_loss_unsup

        f1 = One_Class_GNN_prediction(center, radius, learned_representations, g, 'val', True)['macro avg']['f1-score']

        # Compute gradients
        loss.backward()

        # Tune parameters
        optimizer.step()

        stop, best_embeddings, best_radius, best_center, best_epoch = stopper.step(f1, loss, epoch, radius, center,
                                                                                   learned_representations)

        embeddings.append(learned_representations)
        losses_ocl.append(loss_ocl)
        losses_rec.append(recon_loss_unsup)
        losses.append(loss)
        accuracies.append(f1)

        if stop:
            break

    dic_results = One_Class_GNN_prediction(best_center, best_radius, best_embeddings, g, 'val', True)

    return dic_results

def train_parameters(l_graphs, patience, hidden1, hidden2, r, lr, file_name, pr, epochs):
    multi_task = patience / 2
    l_param = str(hidden1) + '_' + str(hidden2) + '_' + str(r) + '_' + str(lr) + '_' + str(patience)
    metrics = init_metrics()
    for g in l_graphs:
        start = time.time()
        values = train_olga(g, hidden1, hidden2, patience, lr, r, multi_task,epochs)
        end = time.time()
        time_ = end - start
        values['time'] = time_
        save_values(metrics, values)

    write_results(metrics, file_name, l_param, pr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OLGA')
    
    register_data_args(parser)
    
    parser.add_argument("--k", type=str, default='k=1', help="k from graph modeling")
    
    parser.add_argument("--h1", type=int, default=48, help="neurons from first hidden layer")
    
    parser.add_argument("--h2", type=int, default=2, help="neurons from second hidden layer")
    
    parser.add_argument("--h2", type=int, default=2, help="neurons from second hidden layer")
    
    parser.add_argument("--radius", type=float, default=0.5, help="hypershpere radius")
    
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    
    parser.add_argument("--patience", type=int, default=300, help="patience for early stopping")
    
    parser.add_argument("--n-epochs", type=int, default=5000, help="training epochs")
    
    parser.add_argument("--dataset", type=str, default="food", help="dataset")
    
    args = parser.parse_args()

    file_name = 'OLGA.csv' 

    pt = '../datasets/'
    basepath = Path(pt)
    datasets = basepath.iterdir()

    l_graphs = []
    for fold in range(10):
        path = pt + dataset + '/' + args.k + '/' + dataset + '_' + args.k + '_fold=' + str(fold) + '.gpickle'
        graph = nx.read_gpickle(path)
        l_graphs.append(graph)

    # seeds
    seed = 81
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

	pr = '../results/' + dataset + '_' + args.k + '_'

	train_parameters(l_graphs, args.patience, args.h1, args.h2, args.radius, args.lr, file_name, pr, args.n_epochs)
