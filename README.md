# BoRA: Bayesian Hierarchical Low-Rank Adaptation for Multi-task Large Language Models
## Abstract
This paper introduces Bayesian Hierarchical Low-Rank Adaptation (BoRA), a novel method for fine-tuning multi-task Large Language Models (LLMs). 
Current fine-tuning approaches, such as Low-Rank Adaptation (LoRA), perform exceptionally well in reducing training parameters and memory usage but face limitations when applied to multiple similar tasks.
Practitioners usually have to choose between training separate models for each task or a single model for all tasks, both of which come with trade-offs in specialization and data utilization.

BoRA addresses these trade-offs by leveraging a Bayesian hierarchical model that allows tasks to share information through global hierarchical priors. 
This enables tasks with limited data to benefit from the overall structure derived from related tasks while allowing tasks with more data to specialize. 
Our experimental results show that BoRA outperforms both individual and unified model approaches, achieving lower perplexity and better generalization across tasks. 
This method provides a scalable and efficient solution for multi-task LLM fine-tuning, with significant practical implications for diverse applications.
