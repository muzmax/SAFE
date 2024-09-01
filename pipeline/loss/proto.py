import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import numpy as np


def one_hot(targets, num_classes, smoothing=0):
        off_value = smoothing / num_classes
        on_value = 1. - smoothing + off_value
        targets = targets.long().view(-1, 1)
        return torch.full((len(targets), num_classes), off_value).scatter_(1, targets, on_value)

class ProtoLoss(nn.Module):
    def __init__(self, nproto, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1, device = 'cuda'):

        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)
        self.nproto = nproto
        self.student_temp = student_temp
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp))

        self.proto_labels = one_hot(torch.tensor([i for i in range(nproto)]), nproto) # to compute soft 1-NN
        self.proto_labels = self.proto_labels.to(device)

    def snn(self, query, supports, temp=1):
        """ Soft Nearest Neighbours similarity classifier """
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)
        return self.softmax(query @ supports.T / temp) @ self.proto_labels

    def forward(self, student_output, teacher_output, prototypes, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        nb_augment = int(student_output.shape[0]/teacher_output.shape[0])
        probs = self.snn(student_output, prototypes, temp=self.student_temp)
        with torch.no_grad():
            
            targets = self.snn(teacher_output, prototypes, temp = self.teacher_temp_schedule[epoch])
            # targets = torch.cat([targets for _ in range(self.ncrops_student)], dim=0)
            targets = torch.cat([targets for _ in range(nb_augment)], dim=0)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))
        
        avg_probs = torch.mean(probs, dim=0)
        rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + log(float(len(avg_probs)))

        return loss, rloss

    
