# Self-Learning-AI-Metalearning
This repository contains the **research paper, code, and results** for my fun project on **self-improving AI systems**.  
The work explores how AI can **learn to learn more effectively**, inspired by Darwinian evolution and meta-learning principles.

---

## The paper demonstrates:

Recursive self-improvement using meta-learning

Up to 40–50% performance gains over standard methods

S-shaped improvement curves and analysis of failure modes

Ethical and safety considerations for self-improving AI

---

## System Overview:
The self-improving AI system consists of four main components:

Base Learner – Learns tasks (3-layer neural network)

Meta-Learner – Learns how to learn, updating strategies

Evaluator – Measures task performance

Strategy Modifier – Adjusts learning parameters (learning rate, steps)

---

## Workflow:
Sample a task

Base learner adapts

Evaluate performance

Update meta-learner and modify strategy

Repeat recursively

---

## Installation
Clone the repo:

bash
Copy code
git clone https://github.com/Nabz/self-learning-ai-metalearning.git
cd self-improving-ai
Create a Python virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate   # Linux / macOS
.\venv\Scripts\activate  # Windows
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Dependencies:

torch – neural network training

numpy – numerical operations

matplotlib – plotting

---

## Usage:
1. Train the Self-Improving System
bash
Copy code
python code/train.py
Trains on synthetic few-shot tasks

Updates meta-learning strategy dynamically

Saves training progress

2. Visualize Learning Curves
bash
Copy code
python code/utils.py
Generates:

results/learning_curves.png

results/failure_modes.png

These visualize:

S-shaped improvement over iterations

Oscillations and local optima during adaptation

3. Compile the Paper
bash
Copy code
cd paper
pdflatex paper.tex
Generates paper.pdf with all figures and references.

---

## Results
### 1. Few-Shot Classification Accuracy
The self-improving system was evaluated on 5-way 1-shot classification tasks:

Method	Accuracy	Improvement vs Baseline
Baseline	58.3% ± 3.2%	–
Static Meta-Learning	72.6% ± 2.1%	+24.5%
Self-Improving AI	87.1% ± 1.8%	+49.4%

##### Observations:

The self-improving system reaches higher accuracy faster.

Learning curves show S-shaped improvement typical of meta-learning adaptation.

### 2. Training Efficiency
Method	Steps to 80% Accuracy	Time Savings
Baseline	12.4 ± 2.1	–
Static Meta	5.8 ± 0.9	53%
Self-Improving	3.2 ± 0.7	74%

##### Observations:

Recursive self-improvement reduces the number of iterations required.

Strategy modification (adaptive learning rate & steps) accelerates convergence.

### 3. Visual Analysis
Learning Curves:

Shows accuracy progression over iterations.

Self-improving system consistently outperforms baseline and static meta-learning.

Failure Modes:

Demonstrates oscillations and local optima during adaptation.

Useful for debugging and improving meta-learning strategies.

### 4. Summary
Achieves ~87% accuracy on few-shot tasks

Reduces training steps by ~74%

Adaptive meta-learning enables faster, more stable learning

Figures highlight both success and potential pitfalls
