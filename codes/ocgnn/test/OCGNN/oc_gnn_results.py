import numpy as np
import time
from easydict import EasyDict as edict
import dgl
import main
import networkx as nx
from pathlib import Path
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt 

import warnings
warnings.filterwarnings('ignore')
def init_metrics():
    metrics = {
        '1': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        '0': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'macro avg': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'weighted avg': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        'accuracy': [],
        'time': []
    }
    return metrics


def save_values(metrics, values):
    for key in metrics.keys():
      if key == 'accuracy' or key == 'time':
        metrics[key].append(values[key])
      else:
        for key2 in metrics[key].keys():
          metrics[key][key2].append(values[key][key2])


def write_results(metrics, file_name, line_parameters, path):
    if not Path(path + file_name).is_file():
        file_ = open(path + file_name, 'w')
        string = 'Parameters'
        for key in metrics.keys():
            if key == 'accuracy' or key == 'time':
              string += ';' + key + '-mean;' + key + '-std'
            else:
              for key2 in metrics[key].keys():
                string += ';' + key + '_' + key2 + '-mean;' + key + '_' + key2 + '-std'

        string += '\n'
        file_.write(string)
        file_.close()

    file_ = open(path + file_name, 'a')
    string = line_parameters

    for key in metrics.keys():
      if key == 'accuracy' or key == 'time':
        string += ';' + str(np.mean(metrics[key])) + ';' + str(np.std(metrics[key]))
      else:
        for key2 in metrics[key].keys():
          string += ';' + str(np.mean(metrics[key][key2])) + ';' + str(np.std(metrics[key][key2]))

    string += '\n'
    file_.write(string)
    file_.close()

class OneClassGraphDataset(object):
  def __init__(self, graph_networkx, graph_dgl):
    self.labels = graph_dgl.ndata['label'].numpy()
    self.features = graph_dgl.ndata['features'].numpy()
    self.train = graph_dgl.ndata['train'].numpy()
    self.val = graph_dgl.ndata['val'].numpy()
    self.test = graph_dgl.ndata['test'].numpy()
    self.num_labels = 2
    self.graph = graph_networkx

def create_args(dataset, nu, module, lr, patience, n_hidden, k):
  args = edict({"dataset": dataset,
              "dropout": 0.5,
              "nu": nu,
              "gpu": 0,
              "seed": 81,
              "module": module,
              'n_worker': 1,
              'batch_size': 32,
              "lr": lr,
              "normal_class": 1,
              "n_epochs": 500,
              "n_hidden": n_hidden,
              "n_layers": 2,
              "weight_decay": 0.0005,
              'early_stop': True,
              "self_loop": False,
              "norm": False,
              "patience": patience,
              "k": k})

  line_parameters = str(nu) + '_' + str(lr) + '_' + str(patience) + '_' + str(n_hidden)

  return args, line_parameters

if __name__ == '__main__':
  pt = '../../../../datasets/'
  path_results = '../../../../results_test/Low Dim/'
  for dataset in ['relevant_reviews', 'food', 'TUANDROMD']:
    print(dataset)
    for method in ['OC-GCN', 'OC-GAT', 'OC-GraphSAGE']:
      print(method)
      path = pt + dataset + '/k=1/' + dataset + '_k=1_fold=0.gpickle'
      graph = nx.read_gpickle(path)
      graph_dgl = dgl.from_networkx(graph, node_attrs=['features', 'label', 'train', 'val', 'test'], edge_attrs=None, edge_id_attr_name=None)
      g_nt = OneClassGraphDataset(graph,graph_dgl)

      module = method.split('-')[1].split('.')[0]

      df = pd.read_csv(path_results + dataset + '_k=1_' + method + '.csv', sep=';')

      df = df.sort_values(by='macro avg_f1-score-mean', ascending=False)

      for index,row in df.iterrows():
        parts = row['Parameters'].split('_')
        nu = float(parts[0])
        lr = float(parts[1])
        patience = int(parts[2])
        n_hidden = int(parts[3])
        if n_hidden == 2:
          break

        args, line_parameters = create_args(dataset, nu, module, lr, patience, n_hidden, '1')

        values, embs = main.main(args, g_nt)
        colors = []
        for node in graph:
          if graph.nodes[node]['label'] == 1:
            colors.append('#0074ff')
          else:
            colors.append('#00ff5c')
          
        matplotlib.rcParams.update({'font.size': 22})
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot()
        ax.clear()
        ax.scatter(embs[:,0], embs[:,1], s=100, c=colors, cmap="hsv")
        plt.savefig('../' + dataset + '_' + method + '.png')
        plt.close()

