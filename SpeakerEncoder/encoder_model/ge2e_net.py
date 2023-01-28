from scipy.interpolate import interp1d
import sklearn.metrics as metrics
from torch.nn.utils import clip_grad_norm_
from scipy.optimize import brentq

import torch
from torch import nn
import numpy as np
import torch




class GE2Enet(nn.Module):
    def __init__(self,device,activation_function=None):
        super().__init__()
        self.device=torch.device(device)  

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.]))
        self.similarity_bias = nn.Parameter(torch.tensor([-5.]))

        self.similarity_weight.requires_grad_().retain_grad()
        self.similarity_bias.requires_grad_().retain_grad()

   
        self.loss_fn = nn.CrossEntropyLoss()


        self.activation_function=activation_function

        self.cosine_sim=torch.nn.CosineSimilarity(dim=-1)
        self.to(self.device)

    
    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)

    def compare_utterances(self, utterances1, utterances2,use_weight_and_bias=True):
        with torch.no_grad():
            avg_emb1=self.forward(utterances1) #Compute embeddings, batched or not
            
            if type(utterances1)==list:
                avg_emb1=torch.mean(avg_emb1, dim=0, keepdim=True)
                avg_emb1 = avg_emb1 / (torch.norm(avg_emb1, dim=-1, keepdim=True))
            else:
                avg_emb1=avg_emb1.squeeze(0) #If not batched remove a dimension after forward

            avg_emb2=self.forward(utterances2) #Equel for second set of utterances
            if type(utterances2)==list:
                
                avg_emb2=torch.mean(avg_emb2, dim=0, keepdim=True)
                avg_emb2 = avg_emb2 / (torch.norm(avg_emb2, dim=-1, keepdim=True))   
            else:
                avg_emb2=avg_emb2.squeeze(0)

    
            if not use_weight_and_bias:
                if self.activation_function:
                    return self.activation_function(self.cosine_sim(avg_emb1,avg_emb2))
                return self.cosine_sim(avg_emb1,avg_emb2)

            if self.activation_function:
                return self.activation_function(
                        self.similarity_weight*self.cosine_sim(avg_emb1,avg_emb2)+self.similarity_bias 
                        )#compute similarity between averaged (if batched) or single embeddings
            return self.similarity_weight*self.cosine_sim(avg_emb1,avg_emb2)+self.similarity_bias 


    

    
    
    def find_optimal_cutoff(self, fpr, tpr, thresholds):
        tf_values = [tpr[i] - (1 - fpr[i]) for i in range(len(tpr))]
        optimal_threshold = min(tf_values, key=lambda x: abs(x - 0))
        return [thresholds[tf_values.index(optimal_threshold)]]

        


    
    def similarity_matrix(self,output_of_model,speakers_per_batch,utterances_per_speaker):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        embeds=output_of_model.view((speakers_per_batch,utterances_per_speaker,-1))
        
  
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = ((embeds[mask] * centroids_incl[j])/(torch.norm(embeds[mask], dim=-1, keepdim=True) + 1e-5)).sum(dim=2) 
            sim_matrix[j, :, j] = ((embeds[j] * centroids_excl[j])/ (torch.norm(embeds[j], dim=-1, keepdim=True) + 1e-5)).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        if self.activation_function:
            return self.activation_function(sim_matrix)
        return sim_matrix

    def get_ground(self,speakers_per_batch,utterances_per_speaker):
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
        labels = np.array([inv_argmax(i) for i in ground_truth])
        return ground_truth,labels
    
    def loss(self, output_of_model,speakers_per_batch,utterances_per_speaker):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """

        # Loss
        sim_matrix= self.similarity_matrix(output_of_model,speakers_per_batch,utterances_per_speaker)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth,labels = self.get_ground(speakers_per_batch,utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.device)
        loss = self.loss_fn(sim_matrix, target)
        
        # EER (not backpropagated)
        with torch.no_grad():
            preds = sim_matrix.detach().cpu().numpy()
            eer,_,_,_=self.compute_eer(labels,preds)
            
        return loss, eer
    
    def compute_eer(self,labels,preds):
        # Snippet from https://yangcha.github.io/EER-ROC/
        fpr, tpr, thresholds = metrics.roc_curve(labels.flatten(), preds.flatten())           
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

        return eer,fpr,tpr,thresholds





