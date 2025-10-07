import torch
import torch.nn as nn
import numpy as np
from base_learner import BaseLearner

class SelfImprovingSystem:
    def __init__(self):
        self.model = BaseLearner()
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.adaptation_lr = 0.01
        self.adaptation_steps = 5
        self.performance_history = []
    
    def adapt(self, x_support, y_support):
        for _ in range(self.adaptation_steps):
            pred = self.model(x_support)
            loss = nn.CrossEntropyLoss()(pred, y_support)
            grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
            with torch.no_grad():
                for param, grad in zip(self.model.parameters(), grads):
                    param.sub_(self.adaptation_lr * grad)
    
    def train_step(self, tasks):
        meta_loss = 0
        for x_support, y_support, x_query, y_query in tasks:
            self.adapt(x_support, y_support)
            pred = self.model(x_query)
            loss = nn.CrossEntropyLoss()(pred, y_query)
            meta_loss += loss
        meta_loss /= len(tasks)
        self.optimiser.zero_grad()
        meta_loss.backward()
        self.optimiser.step()
        return meta_loss.item()
    
    def improve_strategy(self):
        if len(self.performance_history) < 10:
            return
        recent = self.performance_history[-10:]
        improvement = recent[-1] - recent[0]
        if improvement < 0.01:
            self.adaptation_lr = min(0.05, self.adaptation_lr*1.05)
            self.adaptation_steps = min(10, self.adaptation_steps+1)
        self.adaptation_lr = max(0.001, self.adaptation_lr)
