import torch    
import numpy as np
import torch.nn.functional as F
    
def loss_function(nu,data_center,outputs,radius=0,mask=None):
    dist,scores=anomaly_score(data_center,outputs,radius,mask)
    loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    return loss,dist,scores

def anomaly_score(data_center,outputs,radius=0,mask= None):
    if mask == None:
        dist = torch.sum((outputs - data_center) ** 2, dim=1)
    else:
        dist = torch.sum((outputs[mask] - data_center) ** 2, dim=1)
    # c=data_center.repeat(outputs[mask].size()[0],1)
    # res=outputs[mask]-c
    # res=torch.mean(res, 1, keepdim=True)
    # dist=torch.diag(torch.mm(res,torch.transpose(res, 0, 1)))

    scores = dist - radius ** 2
    return dist,scores

def init_center(args,input_g,input_feat, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    if args.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % args.gpu)
    n_samples = 0
    c = torch.zeros(args.n_hidden, device=device)

    model.eval()
    with torch.no_grad():

        outputs= model(input_g,input_feat)

        # get the inputs of the batch

        n_samples = outputs.shape[0]
        c =torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    radius=np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    # if radius<0.1:
    #     radius=0.1
    return radius

class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_epoch = None
        self.lowest_loss = None
        self.early_stop = False

    def step(self, acc,loss, model,epoch,path,best_radius,radius):
        score = acc
        cur_loss=loss
        if (self.best_score is None) or (self.lowest_loss is None):
        #if self.lowest_loss is None:
            self.best_score = score
            self.lowest_loss = cur_loss
            self.save_checkpoint(acc,loss,model,path)
        #elif cur_loss > self.lowest_loss:
        elif (score < self.best_score):
            self.counter += 1
            #if self.counter >= 0.8*(self.patience):
              #print(f'Warning: EarlyStopping soon: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
              self.early_stop = True
        else:
            best_radius = radius.cpu().numpy()
            self.best_score = score
            self.lowest_loss = cur_loss
            self.best_epoch = epoch
            self.save_checkpoint(acc,loss,model,path)
            self.counter = 0
        return self.early_stop, best_radius

    def save_checkpoint(self, acc,loss,model,path):
        '''Saves model when validation loss decrease.'''
        #print('model saved. loss={:.4f} AUC={:.4f}'. format(loss,acc))
        torch.save(model, path)