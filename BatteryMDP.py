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

    def __init__( self, prices ):
        self.prices = prices

        # Time params
        self.time = 0                                     # Hours passed since start

        # Battery parameters
        self.capacity = self.BATTERY_CAPACITY             # MWh
        self.max_charge = self.MAX_CHARGE_RATE            # MWh per timestep
        self.max_discharge = self.MAX_DISCHARGE_RATE      # MWh per timestep
        self.efficiency = self.ROUND_TRIP_EFFICIENCY
        self.charge = self.INITIAL_CHARGE                 # initial battery charge

    # Reset the environment to the initial state
    def reset( self ):
        self.time = 0
        self.charge = self.INITIAL_CHARGE
        return self.get_state()
    
    # Return the current state as a numpy array [price, charge_level, hour, day_of_week]
    def get_state( self ):
        price = self.prices[self.time]
        hour = self.time % 24
        day_of_week = (self.time // 24) % 7
        return np.array([price, self.charge / self.capacity, hour, day_of_week])
    
    # Take an action and return the new state, reward, and done flag
    def step( self, action ):

        price = self.prices[self.time]
        reward = 0

        if action == self.CHARGE:
            energy = min(self.max_charge, self.capacity - self.charge)
            self.charge += energy * self.efficiency
            reward = -(price * energy)

        elif action == self.DISCHARGE:
            energy = min(self.max_discharge, self.charge)
            self.charge -= energy
            reward = price * energy * self.efficiency

        #  action == self.HOLD -> hold -> reward = 0

        self.time += 1
        done = self.time >= len(self.prices)
        return self.get_state(), reward, done