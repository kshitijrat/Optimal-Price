# Reinforcement Learning-Based Product Pricing Optimization

This repository implements a Reinforcement Learning (RL) module to determine the ideal product price for the next day. The goal is to maximize sales and conversion rates while exploring new price points to optimize profitability.

---

## Overview

The module uses historical sales data, including price points, conversion rates, profits, and sales figures, to train an RL model. It predicts the ideal price for tomorrow by balancing exploration (testing new price points) and exploitation (leveraging historical data). 

Here’s how it works:

1. **Data-Driven Simulation**: 
   - Uses historical data to simulate sales and conversion rates for different price points.
   - Incorporates time series forecasting for predicted sales to refine decision-making.

2. **Reinforcement Learning Approach**:
   - States: Historical and predicted data such as price, sales, and conversion rates.
   - Actions: Incrementing or decrementing the price to explore new price points.
   - Rewards: Higher rewards are assigned for higher sales, better conversion rates, and profits exceeding historical medians. Punishments are assigned for lower-than-predicted sales.

3. **Exploration and Exploitation**:
   - Encourages exploration of price points beyond historical medians by rewarding higher prices and total sales.
   - Ensures the RL model does not get stuck at suboptimal historical price points.

4. **Forecasting and Prediction**:
   - Uses a forecasting model to predict future sales based on historical trends.
   - RL uses these forecasts to recommend optimized prices for the next day.

---

## Features

- **Dynamic Price Optimization**: Continuously adjusts product prices to maximize sales and profitability.
- **Historical and Predicted Data Utilization**: Leverages past trends and future forecasts for improved decision-making.
- **Customizable Reward System**: Configurable rewards and punishments based on sales, conversion rates, and profits.
- **Exploration vs. Exploitation Balance**: Encourages innovative pricing strategies while leveraging proven price points.

---

## Installation

1. Clone the repository:
   ```bash
   https://github.com/kshitijrat/Optimal-Price.git
   ```

2. Navigate to the project directory:
   ```bash
   cd repo-name
   ```

3. Install the required dependencies:
###To run this project, you need to install the following Python packages:

- **NumPy**: A library for numerical computations in Python.
- **Pandas**: A library for data manipulation and analysis.
- **Gym**: A toolkit for developing and comparing reinforcement learning algorithms.

You can install these dependencies using pip. Run the following command in your terminal:
```bash
pip install numpy pandas gym
   ```
If you are using a Jupyter notebook or an IPython environment, you can also install the packages directly from within the notebook by using the following commands:
```bash
!pip install numpy pandas gym
```

If you are using Anaconda, you can create a new environment and install the required packages as follows:
```bash
conda create -n pricing_env python=3.8
conda activate pricing_env
pip install numpy pandas gym
```
---

## Usage

1. Prepare your historical sales data in the required format:
   - Columns: `Report Date`, `Product Price`, `Organic Conversion Percentage`, `Ad Conversion Percentage`, `Total Profit`, `Total Sales`, `Predicted Sales`.

2. Run the RL module:
   ```bash
   python Problem1.py
   ```

3. The program will output the predicted optimal price for tomorrow.

---

## Input and Output Details

### Input
The model takes historical data with the following columns:
- **Report Date**: Date for which the datapoint belongs.
- **Product Price**: The price at which the product was sold.
- **Organic Conversion Percentage**: Conversion rate for organic sales at this price point.
- **Ad Conversion Percentage**: Conversion rate for ads at this price point.
- **Total Profit**: Total profit made on the day at this price point.
- **Total Sales**: Total sales made on the day at this price point.
- **Predicted Sales**: Forecasted sales for future dates.

### Output
The RL module outputs the predicted optimal price for the next day that maximizes:
- Total Sales
- Organic and Ad Conversion Rates
- Overall Profitability

---

## How It Works

1. **Simulation Environment**: The RL model uses historical data to simulate sales and reward outcomes for different price points.
2. **Reward Function**: Designed to reward:
   - Higher-than-average sales.
   - Higher conversion rates (organic and ad-based).
   - Higher profits compared to historical medians.
   - Exploration of price points above historical medians.
3. **Punishment**: Penalizes lower-than-predicted sales or reduced conversion rates.
4. **Exploration**: Actively explores price points beyond historical ranges by testing incremented price values.
5. **Forecasting**: Uses the predicted sales column to refine pricing decisions for the next day.

---

## Example

Input Historical Data:
| Report Date | Product Price | Organic Conversion % | Ad Conversion % | Total Profit | Total Sales | Predicted Sales |
|-------------|---------------|-----------------------|------------------|--------------|-------------|-----------------|
| 2025-02-01  | 14.0          | 12%                  | 10%              | 200          | 100         | 105             |
| 2025-02-02  | 16.0          | 15%                  | 12%              | 250          | 120         | 125             |

Predicted Output:
```
Predicted Optimal Price: $16.5
```

---


## Contact

For any queries or suggestions, feel free to contact:

- **Name**: Kshitij Ratnawat
- **Email**: [kshitijratnawat@gmail.com]
- **LinkedIn**: [(https://www.linkedin.com/in/kshitij-ratnawat/)]

