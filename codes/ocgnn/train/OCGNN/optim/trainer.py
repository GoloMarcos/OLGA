import time
import numpy as np
import torch
import logging
#from dgl.contrib.sampling.sampler import NeighborSampler
# import torch.nn as nn
# import torch.nn.functional as F



from optim.loss import loss_function,init_center,get_radius,EarlyStopping

from utils import fixed_graph_evaluate

def train(args,logger,data,model,path):
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)
    checkpoints_path=path

    # logging.basicConfig(filename=f"./log/{args.dataset}+OC-{args.module}.log",filemode="a",format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",level=logging.INFO)
    # logger=logging.getLogger('OCGNN')
    #loss_fcn = torch.nn.CrossEntropyLoss()
    # use optimizer AdamW
    logger.info('Start training')
    logger.info(f'dropout:{args.dropout}, nu:{args.nu},seed:{args.seed},lr:{args.lr},self-loop:{args.self_loop},norm:{args.norm}')

    logger.info(f'n-epochs:{args.n_epochs}, n-hidden:{args.n_hidden},n-layers:{args.n_layers},weight-decay:{args.weight_decay}')

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    if args.early_stop:
        stopper = EarlyStopping(patience=args.patience)
    # initialize data center

    input_feat=data['features']
    input_g=data['g']

    data_center= init_center(args,input_g,input_feat, model)
    radius=torch.tensor(0, device=device)# radius R initialized with 0 by default.


    #创立矩阵以存储结果曲线
    arr_epoch=np.arange(args.n_epochs)
    arr_loss=np.zeros(args.n_epochs)
    arr_valauc=np.zeros(args.n_epochs)
    arr_testauc=np.zeros(args.n_epochs)

    best_radius = 0
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        #model.train()
        #if epoch %5 == 0:
        t0 = time.time()
        # forward

        outputs= model(input_g,input_feat)
        #print('model:',args.module)
        #print('output size:',outputs.size())

        loss,dist,_=loss_function(args.nu, data_center,outputs,radius,data['train_mask'])
        #保存训练loss
        arr_loss[epoch]=loss.item()
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch>= 3:
            dur.append(time.time() - t0)

        radius.data=torch.tensor(get_radius(dist, args.nu), device=device)

        auc,ap,f1,acc,precision,recall,val_loss, dic = fixed_graph_evaluate(args,checkpoints_path, model, data_center,data,radius,data['val_mask'])

        # print("Epoch {:05d} | Time(s) {:.4f} | Train Loss {:.4f} | Val Loss {:.4f} | Val AUROC {:.4f} | "
        #       "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item()*100,
        #                                     val_loss.item()*3800, auc, data['n_edges'] / np.mean(dur) / 1000))
        # print('F1 de interesse: ' + str(f1_interest))
        # print('Prec interesse: ' + str(pre_interest))
        # print('Rec interest: ' + str(rec_interest))
        # print(f'Val f1:{round(f1,4)},acc:{round(acc,4)},pre:{round(precision,4)},recall:{round(recall,4)}')
        if args.early_stop:
            stop, best_radius = stopper.step(dic['macro avg']['f1-score'],val_loss.item(), model,epoch,checkpoints_path, best_radius,radius)
            if stop:
                break

    if args.early_stop:
        model = torch.load(checkpoints_path)
        best_radius = torch.tensor(best_radius, device=device)
    else:
        best_radius = radius

    auc,ap,f1,acc,precision,recall,val_loss, dic_metrics = fixed_graph_evaluate(args,checkpoints_path, model, data_center,data,best_radius,data['val_mask'])

    return dic_metrics

