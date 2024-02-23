import numpy as np
from pathlib import Path

def init_metrics():
    metrics = {
        '1': {
            'precision': [],
            'recall': [],
            'f1-score': []
        },
        '-1': {
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
