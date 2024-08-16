import numpy as np
import torch
import matplotlib.pyplot as plt

from DRL_Agent_Memory import ReplayMemory, Transition
from DRL_DQN_Agent import DQNAgent, DuellingDQN
from DRL_Env import SingleAssetTradingEnvironment
from DRL_Feature_Generator import DataGetter
from DRL_Global_Params import *

# Cryptocurrency Tickers
asset_codes = ["BTC-USD"]

## Training and Testing Environments
assets = [DataGetter(a, start_date="2019-04-01", end_date="2022-04-01") for a in asset_codes]
test_assets = [DataGetter(a, start_date="2019-04-01", end_date="2023-04-01", freq="1d") for a in asset_codes]
envs = [SingleAssetTradingEnvironment(a) for a in assets]
test_envs = [SingleAssetTradingEnvironment(a) for a in test_assets]

# Agent
memory = ReplayMemory()
agent = DQNAgent(actor_net=DuellingDQN, memory=memory)

# Main training loop
N_EPISODES = 20  # No of episodes/epochs
scores = []
eps = EPS_START
act_dict = {0: -1, 1: 1, 2: 0}

te_score_min = -np.Inf
episode_rewards = []  # List to store rewards for each episode

# Lists to store running rewards and initial and running capital values
running_rewards = []
initial_capital_values = []
running_capital_values = []

for episode in range(1, 1 + N_EPISODES):
    counter = 0
    episode_score = 0
    episode_score2 = 0
    test_score = 0
    test_score2 = 0
    test_actions = []
    test_states = []
    test_capital = []

    for env in envs:
        score = 0
        state = env.reset()
        state = state.reshape(-1, STATE_SPACE)
        while True:
            actions = agent.act(state, eps)
            action = act_dict[actions]
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(-1, STATE_SPACE)

            t = Transition(state, actions, reward, next_state, done)
            agent.memory.store(t)
            agent.learn()

            state = next_state
            score += reward
            counter += 1
            if done:
                break

        episode_score += score
        episode_score2 += (env.store['running_capital'][-1] - env.store['running_capital'][0])

    scores.append(episode_score)
    eps = max(EPS_END, EPS_DECAY * eps)

    for i, test_env in enumerate(test_envs):
        state = test_env.reset()
        done = False
        score_te = 0
        scores_te = [score_te]

        while True:
            actions = agent.act(state)
            action = act_dict[actions]
            test_states.append(state)
            next_state, reward, done, _ = test_env.step(action)
            next_state = next_state.reshape(-1, STATE_SPACE)
            state = next_state
            score_te += reward
            scores_te.append(score_te)
            if done:
                break

        test_score += score_te
        test_score2 += (test_env.store['running_capital'][-1] - test_env.store['running_capital'][0])
        test_capital.append(env.store["running_capital"])
        test_actions.append(env.store["action_store"])

    if test_score > te_score_min:
        te_score_min = test_score
        torch.save(agent.actor_online.state_dict(), "online.pt")
        torch.save(agent.actor_target.state_dict(), "target.pt")

    episode_rewards.append(episode_score)  # Store episode reward
    
    running_rewards.append(sum(scores))  # Store the running reward for the epoch
    
    # Store initial and running capital values
    initial_capital_values.append(env.store['running_capital'][0])
    running_capital_values.append(env.store['running_capital'][-1])

    print(f"Episode: {episode}, Train Score: {episode_score:.5f}, Validation Score: {test_score:.5f}")
    print(f"Episode: {episode}, Train Value: ${episode_score2:.5f}, Validation Value: ${test_score2:.5f}", "\n")

# Plot the reward graph
plt.figure(figsize=(8, 6))
plt.plot(episode_rewards, label='Episode Rewards')
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Reward per Episode')
plt.legend()
plt.grid(True)
plt.show()

# Plot the graph for initial capital to running capital in dollars
plt.figure(figsize=(8, 6))
plt.plot(initial_capital_values, label='Initial Capital ($)')
plt.plot(running_capital_values, label='Running Capital ($)')
plt.xlabel('Episodes')
plt.ylabel('Capital ($)')
plt.title('Initial Capital vs Running Capital')
plt.legend()
plt.grid(True)
plt.show()

# Print profit and initial/final capital
total_episodes = len(episode_rewards)
profit = running_capital_values[-1] - initial_capital_values[0]
print(f"Total Episodes: {total_episodes}")
print(f"Initial Capital: ${initial_capital_values[0]}")
print(f"Final Capital: ${running_capital_values[-1]}")
print(f"Profit: ${running_capital_values[-1] - initial_capital_values[0]}")
