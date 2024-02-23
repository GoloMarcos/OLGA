import numpy as np

def one_class_processing(data,normal_class:int):
  labels, train_idx, test_idx = one_class_labeling(data, data.labels,normal_class)
  return one_class_masking(labels, train_idx, test_idx)


def one_class_labeling(data, labels, normal_class:int):
	
  train_idx=np.where((data.train==1) | ((data.val==1) & (data.labels==normal_class)))[0]
  test_idx = np.where(data.test==1)[0]

  normal_idx=np.where(labels==normal_class)[0]
  abnormal_idx=np.where(labels!=normal_class)[0]

  labels[normal_idx]=0
  labels[abnormal_idx]=1

  return labels.astype("bool"), train_idx, test_idx

#训练集60%正常、验证集15%正常、测试集25%正常，验证集测试集中的正常异常样本1:1
def one_class_masking(labels, train_idx, test_idx):

	train_mask=np.zeros(labels.shape,dtype='bool')
	test_mask=np.zeros(labels.shape,dtype='bool')

	train_mask[train_idx]= 1
	test_mask[test_idx]=1

	return labels, train_mask, test_mask
