import numpy as np
import pandas as pd
import random
import gym
from gym import spaces

data = pd.read_csv('soapnutshistory.csv')
data['Report Date'] = pd.to_datetime(data['Report Date'])
data = data.dropna(subset=["Product Price", "Total Sales"])

class Pricing_Env(gym.Env):
    def __init__(self, data):
        super(Pricing_Env, self).__init__()
        self.data = data
        self.curr = 0

        self.observation_space = spaces.Box(
            low=np.array([data['Product Price'].min(), 0, 0, 0, 0]),
            high=np.array([data['Product Price'].max(), 100, 100, data['Total Sales'].max(), data['Predicted Sales'].max()]),
            dtype=np.float32
        )


        self.action_space = spaces.Discrete(3)

    def reset(self):
        self.curr = random.randint(0, len(self.data) - 1)
        row = self.data.iloc[self.curr]
        return np.array([row['Product Price'], row['Organic Conversion Percentage'], row['Ad Conversion Percentage'], row['Total Sales'], row['Predicted Sales']], dtype=np.float32)

    def step(self, act):
        row = self.data.iloc[self.curr]

        
        if act == 0:  
            new_p = row['Product Price'] - 0.5
        elif act == 1: 
            new_p = row['Product Price']
        else: 
            new_p = row['Product Price'] + 0.5

       
        new_p = max(self.observation_space.low[0], min(new_p, self.observation_space.high[0]))

        close_row = self.data.iloc[(self.data['Product Price'] - new_p).abs().argsort()[:1]]
        next_sale = close_row['Total Sales'].values[0]
        next_pred_sale = close_row['Predicted Sales'].values[0]

       
        reward = (
            next_sale +  
            0.5 * close_row['Organic Conversion Percentage'].values[0] +
            0.5 * close_row['Ad Conversion Percentage'].values[0]
        )

        if next_sale < next_pred_sale:
            reward -= (next_pred_sale - next_sale) * 0.1

        done = self.curr >= len(self.data) - 1

        self.curr = (self.curr + 1) % len(self.data)
        next_state = np.array([
            new_p,
            close_row['Organic Conversion Percentage'].values[0],
            close_row['Ad Conversion Percentage'].values[0],
            next_sale,
            next_pred_sale
        ], dtype=np.float32)

        return next_state, reward, done, {}

env = Pricing_Env(data)
row = data.shape[0]
print("Valid data row's: ", row)
state = env.reset()
for i in range(row):
    act = env.action_space.sample()  
    next_state, reward, done, i = env.step(act)
    print(f"Action(inc or dec): {act}, Next State: {next_state}, Reward: {reward}")
    if done:
        state = env.reset()