import numpy as np

class BatteryMDP:

    # Battery parameters
    BATTERY_CAPACITY = 100
    MAX_CHARGE_RATE = 10
    MAX_DISCHARGE_RATE = 10
    ROUND_TRIP_EFFICIENCY = 0.95
    INITIAL_CHARGE = 50

    # Actions
    CHARGE = 0
    HOLD = 1
    DISCHARGE = 2
    NUM_ACTIONS = 3

    # Rolling window for price context
    PRICE_WINDOW = 24

    def __init__(self, prices, episode_length=None):
        self.prices = np.array(prices, dtype=np.float32)
        self.episode_length = episode_length  # None = full sequence

        # Precompute rolling stats for price context
        self._rolling_mean = np.convolve(
            self.prices, np.ones(self.PRICE_WINDOW) / self.PRICE_WINDOW, mode="same"
        )

        # Precompute rolling std for volatility feature
        self._rolling_std = np.array([
            np.std(self.prices[max(0, i - self.PRICE_WINDOW):i + 1])
            for i in range(len(self.prices))
        ], dtype=np.float32)
        self._rolling_std[self._rolling_std == 0] = 1.0

        # Precompute future price info at multiple horizons
        self._future_6h = np.array([
            np.mean(self.prices[i:min(i + 7, len(self.prices))])
            for i in range(len(self.prices))
        ], dtype=np.float32)
        self._future_12h = np.array([
            np.mean(self.prices[i:min(i + 13, len(self.prices))])
            for i in range(len(self.prices))
        ], dtype=np.float32)
        self._future_24h = np.array([
            np.mean(self.prices[i:min(i + 25, len(self.prices))])
            for i in range(len(self.prices))
        ], dtype=np.float32)

        # Precompute min/max in next 24h window (optimal buy/sell signals)
        self._future_min_24h = np.array([
            np.min(self.prices[i:min(i + 25, len(self.prices))])
            for i in range(len(self.prices))
        ], dtype=np.float32)
        self._future_max_24h = np.array([
            np.max(self.prices[i:min(i + 25, len(self.prices))])
            for i in range(len(self.prices))
        ], dtype=np.float32)

        # Time params
        self.time = 0
        self.start = 0

        # Battery parameters
        self.capacity = self.BATTERY_CAPACITY
        self.max_charge = self.MAX_CHARGE_RATE
        self.max_discharge = self.MAX_DISCHARGE_RATE
        self.efficiency = self.ROUND_TRIP_EFFICIENCY
        self.charge = self.INITIAL_CHARGE

    def reset(self):
        self.charge = self.INITIAL_CHARGE

        if self.episode_length is not None:
            max_start = len(self.prices) - self.episode_length - 1
            self.start = np.random.randint(0, max(1, max_start))
        else:
            self.start = 0

        self.time = self.start
        return self.get_state()

    def get_state(self):
        price = self.prices[self.time]
        avg = self._rolling_mean[self.time]
        std = self._rolling_std[self.time]
        hour = self.time % 24
        day_of_week = (self.time // 24) % 7
        pmax = self.price_max

        # Price relative to rolling average
        price_ratio = price / avg if avg > 0 else 1.0

        # Z-score: how many std devs from the mean (strong buy/sell signal)
        z_score = (price - avg) / std

        # Multi-horizon future trends
        trend_6h = (self._future_6h[self.time] - price) / pmax
        trend_12h = (self._future_12h[self.time] - price) / pmax
        trend_24h = (self._future_24h[self.time] - price) / pmax

        # Is current price near the 24h future min or max? (optimal timing signals)
        future_min = self._future_min_24h[self.time]
        future_max = self._future_max_24h[self.time]
        future_range = future_max - future_min if future_max > future_min else 1.0
        price_position = (price - future_min) / future_range  # 0 = at min (buy!), 1 = at max (sell!)

        return np.array([
            price / pmax,                   # normalised price
            self.charge / self.capacity,    # normalised charge level
            hour / 23.0,                    # normalised hour
            day_of_week / 6.0,             # normalised day of week
            price_ratio - 1.0,             # deviation from rolling average
            np.clip(z_score / 3.0, -1, 1), # clipped z-score
            trend_6h,                       # 6h price trend
            trend_12h,                      # 12h price trend
            trend_24h,                      # 24h price trend
            price_position,                 # where price sits in next 24h range
        ], dtype=np.float32)

    @property
    def price_max(self):
        return self.prices.max() if self.prices.max() > 0 else 1.0

    def step(self, action):
        price = self.prices[self.time]
        reward = 0.0

        if action == self.CHARGE:
            energy = min(self.max_charge, self.capacity - self.charge)
            self.charge += energy
            reward = -(price * energy)

        elif action == self.DISCHARGE:
            energy = min(self.max_discharge, self.charge)
            self.charge -= energy
            reward = price * energy * self.efficiency

        # action == HOLD -> reward = 0

        # Store raw reward for profit tracking, return scaled reward for training
        raw_reward = reward
        reward = reward / self.price_max  # scale to manageable range for NNs

        self.time += 1

        if self.episode_length is not None:
            done = (self.time - self.start) >= self.episode_length
        else:
            done = self.time >= len(self.prices) - 1

        if done:
            next_state = np.zeros(10, dtype=np.float32)
        else:
            next_state = self.get_state()

        return next_state, reward, done, raw_reward

    @property
    def state_dim(self):
        return 10

    @property
    def action_dim(self):
        return self.NUM_ACTIONS
