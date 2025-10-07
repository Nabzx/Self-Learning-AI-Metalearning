import torch
import numpy as np
from self_improving_system import SelfImprovingSystem

def sample_task(n_way=5, k_shot=1, q_query=15):
    x_support = torch.randn(n_way*k_shot, 784)
    y_support = torch.arange(n_way).repeat_interleave(k_shot)
    x_query = torch.randn(n_way*q_query, 784)
    y_query = torch.arange(n_way).repeat_interleave(q_query)
    return x_support, y_support, x_query, y_query

system = SelfImprovingSystem()
num_iterations = 1000

for iteration in range(num_iterations):
    tasks = [sample_task() for _ in range(4)]
    loss = system.train_step(tasks)
    
    if iteration % 50 == 0:
        accuracy = np.random.uniform(0.6, 0.9)
        system.performance_history.append(accuracy)
        system.improve_strategy()
        print(f"Iteration {iteration}: Loss={loss:.4f}, LR={system.adaptation_lr:.4f}")

print("Training complete!")
