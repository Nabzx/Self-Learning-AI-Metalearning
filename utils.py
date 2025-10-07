import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(histories, labels):
    plt.figure(figsize=(10,6))
    for history, label in zip(histories, labels):
        plt.plot(history, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Learning Curves")
    plt.legend()
    plt.savefig("../results/learning_curves.png")
    plt.show()

def plot_failure_modes():
    x = np.linspace(0,10,100)
    y = np.sin(x) + np.random.randn(100)*0.1
    plt.figure(figsize=(10,6))
    plt.plot(x,y)
    plt.title("Failure Modes")
    plt.xlabel("Iteration")
    plt.ylabel("Performance Oscillations")
    plt.savefig("../results/failure_modes.png")
    plt.show()

if __name__ == "__main__":
    plot_learning_curves([np.random.rand(50), np.random.rand(50)], ["Baseline", "Self-Improving"])
    plot_failure_modes()
