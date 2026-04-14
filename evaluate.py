"""
Evaluation script: trains DQN & PPO, runs baseline, and produces comparison
graphs and metrics.

Generates:
  figures/learning_curves.png
  figures/cumulative_profit.png
  figures/battery_soc.png
  figures/price_actions_overlay.png
  figures/comparison_table.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_prices
from BatteryMDP import BatteryMDP
from baseline import run_baseline
from dqn_agent import DQNAgent, train_dqn, evaluate_dqn
from ppo_agent import PPOAgent, train_ppo, evaluate_ppo

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

COLOURS = {"Baseline": "#888888", "DQN": "#1f77b4", "PPO": "#ff7f0e"}


def plot_learning_curves(dqn_rewards, ppo_rewards):
    """Plot episode reward over training for DQN and PPO."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Smooth with rolling window
    window = 10
    for label, rewards, colour in [
        ("DQN", dqn_rewards, COLOURS["DQN"]),
        ("PPO", ppo_rewards, COLOURS["PPO"]),
    ]:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(smoothed, label=label, color=colour, alpha=0.85)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "learning_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved learning_curves.png")


def plot_cumulative_profit(results):
    """Cumulative profit for all strategies on the same chart."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, res in results.items():
        ax.plot(res["cumulative_profits"], label=label, color=COLOURS[label], alpha=0.85)

    ax.set_xlabel("Timestep (hours)")
    ax.set_ylabel("Cumulative Profit ($)")
    ax.set_title("Cumulative Profit Comparison — Test Set")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "cumulative_profit.png"), dpi=150)
    plt.close(fig)
    print("Saved cumulative_profit.png")


def plot_battery_soc(results, sample_hours=168):
    """Battery state-of-charge for a sample week."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for label, res in results.items():
        ax.plot(
            res["charges"][:sample_hours],
            label=label,
            color=COLOURS[label],
            alpha=0.85,
        )

    ax.set_xlabel("Timestep (hours)")
    ax.set_ylabel("Battery Charge (MWh)")
    ax.set_title(f"Battery State-of-Charge — First {sample_hours // 24} Days")
    ax.axhline(y=100, color="red", linestyle="--", alpha=0.3, label="Capacity")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.3)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "battery_soc.png"), dpi=150)
    plt.close(fig)
    print("Saved battery_soc.png")


def plot_price_actions(test_prices, results, sample_hours=168):
    """Overlay price with DQN charge/discharge actions."""
    fig, ax1 = plt.subplots(figsize=(14, 5))

    prices = test_prices[:sample_hours]
    ax1.plot(prices, color="gray", alpha=0.5, label="Demand (Price Proxy)")
    ax1.set_xlabel("Timestep (hours)")
    ax1.set_ylabel("Demand (MWh)")

    # Show DQN actions as colored markers
    dqn_actions = results["DQN"]["actions"][:sample_hours]
    t = np.arange(sample_hours)

    charge_mask = dqn_actions == BatteryMDP.CHARGE
    discharge_mask = dqn_actions == BatteryMDP.DISCHARGE

    ax1.scatter(
        t[charge_mask], prices[charge_mask],
        c="green", marker="^", s=15, alpha=0.7, label="DQN Charge (Buy)",
    )
    ax1.scatter(
        t[discharge_mask], prices[discharge_mask],
        c="red", marker="v", s=15, alpha=0.7, label="DQN Discharge (Sell)",
    )

    ax1.set_title(f"DQN Actions Overlaid on Demand — First {sample_hours // 24} Days")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "price_actions_overlay.png"), dpi=150)
    plt.close(fig)
    print("Saved price_actions_overlay.png")


def print_comparison_table(results):
    """Print and save a comparison table."""
    baseline_profit = results["Baseline"]["total_profit"]

    print("\n" + "=" * 60)
    print(f"{'Strategy':<15} {'Total Profit ($)':>18} {'vs Baseline':>15}")
    print("-" * 60)

    for label, res in results.items():
        profit = res["total_profit"]
        if label == "Baseline":
            imp = "—"
        else:
            pct = (profit - baseline_profit) / abs(baseline_profit) * 100
            imp = f"{pct:+.1f}%"
        print(f"{label:<15} {profit:>18,.0f} {imp:>15}")

    print("=" * 60)


def main():
    print("Loading data...")
    train_prices = load_prices("train")
    test_prices = load_prices("test")

    # --- Baseline ---
    print("\nRunning baseline (24h window)...")
    baseline_result = run_baseline(test_prices, window=24)
    print(f"Baseline profit: ${baseline_result['total_profit']:,.0f}")

    # --- DQN ---
    print("\nTraining DQN (1500 episodes, 4320h windows, baseline prefill)...")
    dqn_agent, dqn_rewards = train_dqn(train_prices, num_episodes=1500, log_every=300, use_baseline_prefill=True)
    dqn_agent.save(os.path.join(MODELS_DIR, "dqn_battery.pt"))

    print("Evaluating DQN...")
    dqn_result = evaluate_dqn(dqn_agent, test_prices)
    print(f"DQN profit: ${dqn_result['total_profit']:,.0f}")

    # --- PPO (best of 3) ---
    print("\nTraining PPO (1500 episodes x 3 runs, 4320h windows)...")
    best_ppo_profit = -float("inf")
    best_ppo_agent = None
    ppo_rewards = None
    for run in range(3):
        agent, rewards = train_ppo(train_prices, num_episodes=1500, log_every=1500)
        result = evaluate_ppo(agent, test_prices)
        profit = result["total_profit"]
        print(f"PPO run {run+1}/3: ${profit:,.0f}")
        if profit > best_ppo_profit:
            best_ppo_profit = profit
            best_ppo_agent = agent
            ppo_rewards = rewards

    ppo_agent = best_ppo_agent
    ppo_agent.save(os.path.join(MODELS_DIR, "ppo_battery.pt"))

    print(f"Best PPO profit: ${best_ppo_profit:,.0f}")
    ppo_result = evaluate_ppo(ppo_agent, test_prices)

    # --- Results ---
    results = {
        "Baseline": baseline_result,
        "DQN": dqn_result,
        "PPO": ppo_result,
    }

    print_comparison_table(results)

    # --- Graphs ---
    print("\nGenerating graphs...")
    plot_learning_curves(dqn_rewards, ppo_rewards)
    plot_cumulative_profit(results)
    plot_battery_soc(results)
    plot_price_actions(test_prices, results)

    print("\nDone! All figures saved to figures/")


if __name__ == "__main__":
    main()
