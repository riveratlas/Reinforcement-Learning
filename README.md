# State-of-the-Art Reinforcement Learning Models & Algorithms

A comprehensive collection of cutting-edge Reinforcement Learning models and algorithms, organized by categories for easy navigation and reference.

## Table of Contents
- [Value-Based Methods](#value-based-methods)
- [Policy Gradient Methods](#policy-gradient-methods)
- [Actor-Critic Methods](#actor-critic-methods)
- [Model-Based Reinforcement Learning](#model-based-reinforcement-learning)
- [Exploration Techniques](#exploration-techniques)
- [Multi-Agent Reinforcement Learning](#multi-agent-reinforcement-learning)
- [Hierarchical Reinforcement Learning](#hierarchical-reinforcement-learning)
- [Imitation Learning](#imitation-learning)
- [Meta Reinforcement Learning](#meta-reinforcement-learning)
- [Resources](#resources)

## Value-Based Methods

### Deep Q-Network (DQN) - 2015
- **Description**: Combines Q-Learning with deep neural networks to learn policies from high-dimensional sensory inputs.
- **Key Innovations**: Experience replay, fixed Q-targets
- **Benchmarks**: Human-level performance on many Atari 2600 games
- **Paper**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- **Code**: [OpenAI Baselines](https://github.com/openai/baselines)

### Double DQN (DDQN) - 2015
- **Description**: Addresses the overestimation bias in DQN by decoupling action selection and evaluation.
- **Key Innovations**: Double Q-learning for value estimation
- **Paper**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

### Dueling DQN - 2016
- **Description**: Separates the estimation of state values and advantages for better policy evaluation.
- **Key Innovations**: Dueling network architecture
- **Paper**: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

## Policy Gradient Methods

### REINFORCE - 1992
- **Description**: A Monte Carlo policy gradient algorithm that updates parameters in the direction of higher rewards.
- **Key Innovations**: Direct policy optimization
- **Paper**: [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696)

### Proximal Policy Optimization (PPO) - 2017
- **Description**: A policy optimization method that uses a clipped objective function for stable training.
- **Key Innovations**: Clipped surrogate objective, multiple epochs of optimization
- **Benchmarks**: State-of-the-art on many continuous control tasks
- **Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **Code**: [OpenAI Spinning Up](https://spinningup.openai.com/)

## Actor-Critic Methods

### Advantage Actor-Critic (A2C) - 2016
- **Description**: Combines the benefits of policy gradients and value function approximation.
- **Key Innovations**: Advantage function for reduced variance
- **Paper**: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)

### Soft Actor-Critic (SAC) - 2018
- **Description**: An off-policy actor-critic method that optimizes a stochastic policy with entropy regularization.
- **Key Innovations**: Maximum entropy framework, automatic temperature tuning
- **Paper**: [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)

## Model-Based Reinforcement Learning

### Model-Based Policy Optimization (MBPO) - 2019
- **Description**: A model-based RL algorithm that uses an ensemble of learned dynamics models.
- **Key Innovations**: Model-based policy optimization with uncertainty estimation
- **Paper**: [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253)

## Exploration Techniques

### Random Network Distillation (RND) - 2018
- **Description**: Uses prediction errors of a random network as an intrinsic reward signal for exploration.
- **Key Innovations**: Intrinsic curiosity-driven exploration
- **Paper**: [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)

## Multi-Agent Reinforcement Learning

### MADDPG - 2017
- **Description**: A multi-agent actor-critic method that learns policies in continuous action spaces.
- **Key Innovations**: Centralized training with decentralized execution
- **Paper**: [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275)

## Hierarchical Reinforcement Learning

### HIRO - 2018
- **Description**: A hierarchical reinforcement learning method that learns temporal abstractions.
- **Key Innovations**: Data-efficient hierarchical learning
- **Paper**: [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296)

## Imitation Learning

### GAIL - 2016
- **Description**: Uses generative adversarial training for imitation learning.
- **Key Innovations**: Adversarial training for policy matching
- **Paper**: [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476)

## Meta Reinforcement Learning

### MAML - 2017
- **Description**: Model-Agnostic Meta-Learning for fast adaptation to new tasks.
- **Key Innovations**: Meta-learning for RL
- **Paper**: [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)

## Resources

### Learning Materials
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Deep Reinforcement Learning Course (David Silver)](https://www.davidsilver.uk/teaching/)
- [Berkeley CS 285: Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/)

### Frameworks
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [RLlib](https://docs.ray.io/en/latest/rllib/index.html)
- [Dopamine](https://github.com/google/dopamine)

### Environments
- [OpenAI Gym](https://gym.openai.com/)
- [DeepMind Control Suite](https://github.com/deepmind/dm_control)
- [StarCraft II Learning Environment](https://github.com/deepmind/pysc2)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request with updates to algorithms, new papers, or additional resources.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.