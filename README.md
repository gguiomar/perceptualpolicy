# Perceptual Policy Learning

This repository contains implementations of various reinforcement learning agents with different policy network architectures for comparison. The implementation includes Maximum Entropy RL, PPO, TRPO, and supports MLP, RNN, and Transformer policy networks.

## Installation

### Option 1: Using Conda (recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/perceptualpolicy.git
   cd perceptualpolicy
   ```

2. Create a conda environment using the provided environment file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate rl_perceptual
   ```

### Option 2: Using pip

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/perceptualpolicy.git
   cd perceptualpolicy
   ```

2. Create a virtual environment:
   ```bash
   python -m venv rl_env
   ```

3. Activate the virtual environment:
   ```bash
   # On Windows
   rl_env\Scripts\activate
   # On macOS/Linux
   source rl_env/bin/activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Verify Installation

To verify that PyTorch is installed correctly with CUDA support:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"
```

## Project Structure

```
perceptualpolicy/
├── agents/
│   ├── base_agent.py
│   ├── maxent_agent.py
│   ├── policy_networks.py
│   ├── ppo_agent.py
│   └── trpo_agent.py
├── envs/
│   └── gridworld.py
├── utils/
|   └──visualization.py
├── plots/
├── run_example.py
├── run_all.py
├── environment.yml
```

## Running the Simulations

### Example Scenarios

The repository provides an easy-to-use script for testing different scenarios:
```bash
python run_example.py
```

This interactive script allows you to choose from three different examples:
1. Single agent (PPO with MLP)
2. Agent comparison (MaxEnt, PPO, TRPO with MLP)
3. Policy comparison (PPO with MLP, RNN, Transformer)

### Full Comparison Simulation

For a comprehensive comparison of all agents and policy network types:
```bash
python run_all.py
```

You can modify this script to adjust hyperparameters such as:
- Grid size and obstacle configuration
- Number of training episodes
- Learning rates and other agent-specific parameters
- Network architectures

## Available Agents

- **MaxEnt**: Maximum Entropy Reinforcement Learning with temperature-controlled exploration
- **FisherMaxEnt**: MaxEnt variant with Fisher Information Matrix for natural gradient updates
- **PPO**: Proximal Policy Optimization with clipped surrogate objective
- **TRPO**: Trust Region Policy Optimization with KL constraint

## Policy Network Architectures

- **MLP**: Standard multi-layer perceptron with configurable hidden layers
- **RNN**: Recurrent neural network (GRU/LSTM) for sequential decision making
- **Transformer**: Transformer-based policy for capturing long-range dependencies

## Visualization

The code automatically generates various visualizations including:
- Training curves (rewards, losses, entropy/KL)
- Policy heatmaps showing action probabilities
- Trajectory samples showing agent behavior
- State visitation frequency heatmaps

All visualizations are saved to the `plots/` directory.

## Customizing the Environment

The GridWorld environment can be customized with:
```python
env = GridWorld(
    grid_size=10,            # Size of the grid
    start=(0, 0),            # Starting position
    goal=(9, 9),             # Goal position
    max_steps=100,           # Maximum steps per episode
    stochastic=False,        # Whether actions are stochastic
    noise=0.1,               # Noise level for stochastic actions
    add_obstacles=True,      # Whether to include obstacles
    custom_obstacles=None    # Custom obstacle positions
)
```

## Extending the Framework

To add your own agent:
1. Create a new agent class that inherits from `BaseAgent`
2. Implement the `update()` method for your policy update logic
3. Add your agent to the comparison scripts

To add a new policy network:
1. Add your network class to `policy_networks.py`
2. Update the `create_policy_network()` factory function

## License

[Include license information here]

## Citation

If you use this code in your research, please cite:
[Include citation information here]