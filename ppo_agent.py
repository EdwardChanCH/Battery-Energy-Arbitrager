import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from BatteryMDP import BatteryMDP


class ActorCritic(nn.Module):
    """Shared-backbone actor-critic network."""

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    """PPO agent with clipped objective and GAE."""

    def __init__(
        self,
        state_dim=4,
        action_dim=3,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        epochs_per_update=10,
        batch_size=64,
        entropy_coef=0.01,
        value_coef=0.5,
        device=None,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state, eval_mode=False):
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if eval_mode:
                logits, _ = self.model(s)
                return logits.argmax(dim=1).item()
            action, log_prob, value = self.model.get_action(s)
            return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        """Compute Generalised Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, log_probs_old, returns, advantages):
        """Run PPO update for multiple epochs."""
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        log_probs_old_t = torch.FloatTensor(log_probs_old).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Normalise advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n = len(states)
        total_loss = 0.0
        num_updates = 0

        for _ in range(self.epochs_per_update):
            indices = np.arange(n)
            np.random.shuffle(indices)

            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                log_probs_new, values_new, entropy = self.model.evaluate(
                    states_t[idx], actions_t[idx]
                )

                # Policy loss (clipped)
                ratio = torch.exp(log_probs_new - log_probs_old_t[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values_new, returns_t[idx])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        return total_loss / max(num_updates, 1)

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])


def train_ppo(train_prices, num_episodes=500, episode_length=4320, log_every=50):
    """Train PPO on the battery arbitrage environment."""
    env = BatteryMDP(train_prices, episode_length=episode_length)
    agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    episode_rewards = []

    for ep in range(num_episodes):
        state = env.reset()
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []
        total_reward = 0.0

        while True:
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, raw_reward = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)  # scaled reward for training
            values.append(value)
            dones.append(float(done))

            total_reward += raw_reward  # track raw profit
            state = next_state

            if done:
                break

        # Compute GAE and update
        advantages, returns = agent.compute_gae(
            np.array(rewards), np.array(values), np.array(dones)
        )

        agent.update(
            np.array(states),
            np.array(actions),
            np.array(log_probs),
            returns,
            advantages,
        )

        episode_rewards.append(total_reward)

        if (ep + 1) % log_every == 0:
            avg = np.mean(episode_rewards[-log_every:])
            print(f"Episode {ep+1}/{num_episodes}  |  Avg Reward: {avg:,.0f}")

    return agent, episode_rewards


def evaluate_ppo(agent, prices):
    """Evaluate a trained PPO agent on a price series."""
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

    print("Training PPO...")
    agent, rewards = train_ppo(train_prices, num_episodes=300)

    agent.save("models/ppo_battery.pt")

    print("\nEvaluating on test set...")
    result = evaluate_ppo(agent, test_prices)
    print(f"PPO Test Profit: ${result['total_profit']:,.0f}")
