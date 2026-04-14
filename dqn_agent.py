import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from BatteryMDP import BatteryMDP


class QNetwork(nn.Module):
    """Simple feedforward Q-network."""

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """DQN agent with target network and epsilon-greedy exploration."""

    def __init__(
        self,
        state_dim=10,
        action_dim=3,
        lr=1e-3,
        gamma=0.99,
        buffer_size=50_000,
        batch_size=64,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=500_000,
        target_update_freq=1_000,
        device=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_freq = target_update_freq

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    @property
    def epsilon(self):
        """Linearly decay epsilon from eps_start to eps_end."""
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state, eval_mode=False):
        """Epsilon-greedy action selection."""
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_vals = self.q_net(s)
            return q_vals.argmax(dim=1).item()

    def store(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        """One gradient step on a mini-batch from the replay buffer."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        q_values = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1

        # Update target network
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps = ckpt["steps"]


def _prefill_baseline(agent, prices, window=24):
    """Pre-fill replay buffer with baseline experience (imitation learning)."""
    from baseline import rule_based_policy
    env = BatteryMDP(prices)
    state = env.reset()

    for t in range(len(prices) - 1):
        avg = np.mean(prices[max(0, t - window):t + 1])
        action = rule_based_policy(prices[t], avg)
        next_state, reward, done, _ = env.step(action)
        agent.store(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    # Pre-train on baseline experience
    for _ in range(1000):
        agent.update()
    print(f"Pre-filled buffer with {len(agent.buffer)} baseline transitions, ran 1000 updates")


def train_dqn(train_prices, num_episodes=500, episode_length=4320, log_every=50,
              update_every=4, use_baseline_prefill=True):
    """Train DQN on the battery arbitrage environment."""
    env = BatteryMDP(train_prices, episode_length=episode_length)
    agent = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    if use_baseline_prefill:
        _prefill_baseline(agent, train_prices)

    episode_rewards = []
    step_count = 0

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, raw_reward = env.step(action)

            agent.store(state, action, reward, next_state, done)
            step_count += 1
            if step_count % update_every == 0:
                agent.update()

            total_reward += raw_reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)

        if (ep + 1) % log_every == 0:
            avg = np.mean(episode_rewards[-log_every:])
            print(f"Episode {ep+1}/{num_episodes}  |  "
                  f"Avg Reward: {avg:,.0f}  |  Epsilon: {agent.epsilon:.3f}")

    return agent, episode_rewards


def evaluate_dqn(agent, prices):
    """Evaluate a trained DQN agent on a price series."""
    env = BatteryMDP(prices)
    state = env.reset()

    total_profit = 0.0
    profits = []
    charges = []
    actions_taken = []

    while True:
        action = agent.select_action(state, eval_mode=True)
        next_state, reward, done, raw_reward = env.step(action)

        total_profit += raw_reward
        profits.append(total_profit)
        charges.append(env.charge)
        actions_taken.append(action)

        state = next_state
        if done:
            break

    return {
        "total_profit": total_profit,
        "cumulative_profits": np.array(profits),
        "charges": np.array(charges),
        "actions": np.array(actions_taken),
    }


if __name__ == "__main__":
    from data_loader import load_prices

    train_prices = load_prices("train")
    test_prices = load_prices("test")

    print("Training DQN...")
    agent, rewards = train_dqn(train_prices, num_episodes=300)

    agent.save("models/dqn_battery.pt")

    print("\nEvaluating on test set...")
    result = evaluate_dqn(agent, test_prices)
    print(f"DQN Test Profit: ${result['total_profit']:,.0f}")
