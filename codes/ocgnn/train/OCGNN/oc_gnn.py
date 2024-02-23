import numpy as np
import time
from easydict import EasyDict as edict
import torch
import dgl
import main
import networkx as nx
from pathlib import Path
import pandas as pd
import warnings
import sys

warnings.filterwarnings('ignore')

def define_nus():
  nus = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
  return nus

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
              "n_epochs": 5000,
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

	dataset = sys.argv[1]
	
	for k in ['k=1', 'k=2', 'k=3']:
		for module in ['GCN', 'GAT', 'GraphSAGE']:
			file_name = 'OC-' + module + '.csv' #+ '-128.csv'

			pt = '../datasets/'
			path_results = '../results/'

			basepath = Path(pt)
			datasets = basepath.iterdir()

			print('no dataset: ' + dataset)
			l_graphs = []
			for fold in range(10):
				path = pt + dataset + '/' + k + '/' + dataset + '_' + k + '_fold=' + str(fold) + '.gpickle'

				graph = nx.read_gpickle(path)
				graph_dgl = dgl.from_networkx(graph, node_attrs=['features', 'label', 'train', 'val', 'test'], edge_attrs=None, edge_id_attr_name=None)
				g_nt = OneClassGraphDataset(graph,graph_dgl)
				l_graphs.append(g_nt)

			pr = path_results + dataset + '_' + k + '_'

			for nu in define_nus():
				for lr in [0.0001, 0.0005]:
					for patience in [300, 500]:
						for n_hidden in [2,3,128]:
							args, line_parameters = create_args(dataset, nu, module, lr, patience, n_hidden, k)
							metrics = init_metrics()
							for g in l_graphs:
								start = time.time()
								values = main.main(args, g)
								end = time.time()
								time_ = end - start
								values['time'] = time_
								save_values(metrics, values)            
							write_results(metrics, file_name, line_parameters, pr)

