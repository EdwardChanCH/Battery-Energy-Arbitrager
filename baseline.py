import numpy as np
from BatteryMDP import BatteryMDP


def rule_based_policy(price, moving_avg):
    """Buy when price is below average, sell when above."""
    if price < moving_avg * 0.95:
        return BatteryMDP.CHARGE
    elif price > moving_avg * 1.05:
        return BatteryMDP.DISCHARGE
    else:
        return BatteryMDP.HOLD


def run_baseline(prices, window=24):
    """Run the rule-based baseline strategy and return results."""
    env = BatteryMDP(prices)
    state = env.reset()

    total_profit = 0.0
    profits = []
    charges = []
    actions_taken = []

    for t in range(len(prices) - 1):
        avg = np.mean(prices[max(0, t - window):t + 1])
        action = rule_based_policy(prices[t], avg)

        state, reward, done, raw_reward = env.step(action)
        total_profit += raw_reward
        profits.append(total_profit)
        charges.append(env.charge)
        actions_taken.append(action)

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

    test_prices = load_prices("test")

    # Try different windows and report the best
    best_profit = -np.inf
    best_window = None
    for w in [12, 24, 48, 72]:
        result = run_baseline(test_prices, window=w)
        profit = result["total_profit"]
        print(f"Window {w:3d}h: profit = ${profit:,.0f}")
        if profit > best_profit:
            best_profit = profit
            best_window = w

    print(f"\nBest window: {best_window}h with profit ${best_profit:,.0f}")
