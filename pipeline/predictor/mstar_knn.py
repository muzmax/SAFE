import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms 
from torch.utils.data import DataLoader

from ..datasets.load import load_separated_paths
from ..datasets.preprocessing import normalization,ToTensor, amplitude_norm
from ..datasets.datasets import mstar_data
from ..utils import draw_progress_bar
from ..logger import LOGGER

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap


def labels_str2tensor(train_labels,test_labels,device):
    dico = {}
    x = train_labels+test_labels
    count = 0
    for label in x:
        if label not in dico:
            dico[label] = count
            count += 1
    train_labels_t = torch.tensor([dico[i] for i in train_labels],device=device)
    test_labels_t = torch.tensor([dico[i] for i in test_labels],device=device)
    return train_labels_t, test_labels_t

class mstar_knn():
    def __init__(self,norm:list,
                 eval_path:str,
                 labeled_path:str,
                 n_label:int,
                 reduction:bool,
                 red_dim:int,
                 red_type:str,
                 device :str,
                 no_network: bool
                 ) -> None:
        assert len(norm) == 2
        
        if no_network:
            transform = ToTensor()
        else :
            transform = transforms.Compose([normalization(norm[0],norm[1]),ToTensor()])
        
        train_paths,eval_paths,_ = load_separated_paths(dir1=labeled_path,dir2=eval_path,n_label=n_label)
        
        train_data = mstar_data(train_paths,transform)  
        eval_data = mstar_data(eval_paths,transform)  

        self.data_loader_train = DataLoader(
                            train_data,
                            batch_size=1, # images of different size
                            shuffle=False,
                            num_workers=0)
        self.data_loader_eval = DataLoader(
                            eval_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
        self.device = device
        self.reduction = reduction
        self.red_dim = red_dim
        self.red_type = red_type
        self.no_network = no_network

    def red_ft(self, feat_train,feat_eval):
        feat_train_np = feat_train.cpu().data.numpy()
        feat_eval_np = feat_eval.cpu().data.numpy()
        concatenated_features = np.concatenate((feat_train_np, feat_eval_np), axis=0)
        scaler = StandardScaler()
        concatenated_features = scaler.fit_transform(concatenated_features)

        if self.red_type == 'pca':
            pca = PCA(n_components=self.red_dim)
            x_reduced = pca.fit_transform(concatenated_features)
            LOGGER.info('Cumulative explained variation for {} principal components: {}'.format(self.red_dim,np.sum(pca.explained_variance_ratio_)))

        elif self.red_type == 'umap':
            umap_model = umap.UMAP(
                n_neighbors=10,
                min_dist=0.1,
                n_components=self.red_dim,  
                metric='euclidean' # or 'cosine'
            )
            x_reduced = umap_model.fit_transform(concatenated_features)
        
        feat_train_np = x_reduced[:feat_train.shape[0]]
        feat_eval_np = x_reduced[feat_train.shape[0]:]
        feat_train_red = torch.from_numpy(feat_train_np)
        feat_eval_red = torch.from_numpy(feat_eval_np)
        return feat_train_red.to(self.device),feat_eval_red.to(self.device)
        
    @torch.no_grad()
    def extract_features(self,model):

        # Extract features from training set
        count = 1
        total_len = len(self.data_loader_train)
        features_train = None
        LOGGER.info('Exctracting labeled features ({} vectors) ...'.format(total_len))
        for samples,_,_,index in self.data_loader_train:
            draw_progress_bar(count,total=total_len)
            count += 1
            samples = samples.to(self.device,non_blocking=True)
            index = index.to(self.device,non_blocking=True)
            feats = model(samples).clone()
            # init storage feature matrix
            if features_train is None:
                features_train = torch.zeros(total_len, feats.shape[-1])
                features_train = features_train.to(self.device, non_blocking=True)
                # print("Storing features into tensor of shape {}".format(features_train.shape))
            if self.device == 'cuda':
                features_train.index_copy_(0, index, feats)
            else:
                features_train.index_copy_(0, index.cpu(), feats.cpu())
        # Exctract features from evaluation set
        count = 1
        total_len = len(self.data_loader_eval)
        features_eval = None
        LOGGER.info('Exctracting evaluation features ({} vectors) ...'.format(total_len))
        for samples,_,_,index in self.data_loader_eval:
            draw_progress_bar(count,total=total_len)
            count += 1
            samples = samples.to(self.device,non_blocking=True)
            index = index.to(self.device,non_blocking=True)
            feats = model(samples).clone()
            # init storage feature matrix
            if features_eval is None:
                features_eval = torch.zeros(total_len, feats.shape[-1])
                features_eval = features_eval.to(self.device, non_blocking=True)
                # print("Storing features into tensor of shape {}".format(features_eval.shape))
            if self.device == 'cuda':
                features_eval.index_copy_(0, index, feats)
            else:
                features_eval.index_copy_(0, index.cpu(), feats.cpu())

        if self.reduction:
            features_train,features_eval = self.red_ft(features_train,features_eval)
            
        features_train = nn.functional.normalize(features_train, dim=1, p=2)
        features_eval = nn.functional.normalize(features_eval, dim=1, p=2)
        train_labels = self.data_loader_train.dataset.get_all_labels()
        test_labels = self.data_loader_eval.dataset.get_all_labels()
        train_labels,test_labels = labels_str2tensor(train_labels,test_labels,device=self.device)

        return features_train, features_eval, train_labels, test_labels
    
    @torch.no_grad()
    def knn_classifier(self, train_features, train_labels, test_features, test_labels, k, T, num_classes=None):
        if num_classes == None:
            num_classes = torch.max(test_labels).item()+1
        top1, top5, total = 0.0, 0.0, 0
        train_features = train_features.t()
        num_test_images, num_chunks = test_labels.shape[0], 1
        imgs_per_chunk = num_test_images // num_chunks
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
        for idx in range(0, num_test_images, imgs_per_chunk):
            # get the features for test images
            features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
            targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
            batch_size = targets.shape[0]

            # calculate the dot product and compute top-k neighbors
            similarity = torch.mm(features, train_features) # distance between each test feature and all the train features
            distances, indices = similarity.topk(k, largest=True, sorted=True) # get only the top k distances and indices 
            candidates = train_labels.view(1, -1).expand(batch_size, -1) # duplicates labels for process
            retrieved_neighbors = torch.gather(candidates, 1, indices) # for each test feature get the top k label 
            
            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1) # for each test feature get the one hot version of the top k labels 
            
            distances_transform = distances.clone().div_(T).exp_() # softmax with temperature (without the division)
            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),),1,) # get the score for each label (b_sz*num_classes)
        
            _, predictions = probs.sort(1, True) # sort labels by descencing score 

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
        
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            if k>=5 and num_classes>5:# top5 does not make sense if k or num_classes < 5
                top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  
            total += targets.size(0)
        
        top1 = top1 * 100.0 / total
        if k>=5 and num_classes>5:
            top5 = top5 * 100.0 / total
            return top1, top5
        return top1,None 

    def __call__(self, model : nn.modules, nb_knn : list, temperature=0.07):
        results = {}
        features_train, features_eval, train_labels, test_labels = self.extract_features(model)
        for k in nb_knn:
            top1, top5 = self.knn_classifier(features_train, train_labels, features_eval, test_labels, k, temperature)
            results['{}-NN - top 1'.format(k)] = top1 
            if top5 != None:
                results['{}-NN - top 5'.format(k)] = top5 
        return results
    



        
