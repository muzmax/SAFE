import torch
import torch.nn.functional as F
import torch.nn as nn

# Transform the true "Long" labels to softlabels. The confidence of the gt class is 
#  1-smoothing, and the rest of the probability (i.e. smoothing) is uniformly distributed
#  across the non-gt classes. Note, this is slightly different than standard smoothing
#  notation.  

def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
	"""
	if smoothing == 0, it's one-hot method
	if 0 < smoothing < 1, it's smooth method
	"""
	assert 0 <= smoothing < 1
	confidence = 1.0 - smoothing
	label_shape = torch.Size((true_labels.size(0), classes))
	with torch.no_grad():	
		true_dist = torch.empty(size=label_shape, device=true_labels.device)
		true_dist.fill_(smoothing / (classes - 1))
		true_dist.scatter_(1, true_labels.argmax(dim=1, keepdim=True), confidence)
	return true_dist.float()

def xent_with_soft_targets(logit_preds, targets):
	logsmax = F.log_softmax(logit_preds, dim=1)
	batch_loss = targets * logsmax
	batch_loss =  -1*batch_loss.sum(dim=1)
	return batch_loss.mean()    


class SoftmaxSmooth(nn.Module):
	def __init__(self,n_labels,smoothing=0.1) -> None:
		super().__init__()
		self.n_labels = n_labels
		self.smoothing = smoothing
	def forward(self,outputs,labels):
		soft_labels = smooth_one_hot(labels,self.n_labels,self.smoothing)
		return xent_with_soft_targets(outputs,soft_labels)
	
        