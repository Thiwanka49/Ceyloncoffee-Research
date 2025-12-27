import pandas as pd
import numpy as np
import os

def generate_synthetic_data(start_date='2015-01-01', end_date='2025-12-31'):
    """
    Generates synthetic monthly data for Sri Lankan coffee export price and demand.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='M')
    n_months = len(dates)

    # --- Seed control for reproducibility ---
    np.random.seed(42)

    # --- Arabica Price Generation (Higher value, volatile) ---
    # Geometric Brownian Motion-like path
    # Base price around $12, drifting to $15 then dropping, then recovering
    arabica_prices = []
    price = 12.0
    for i, date in enumerate(dates):
        # Trend components
        trend = 0.0
        if date.year == 2020:
             trend = 0.5 # Spike
        elif date.year == 2021:
             trend = -0.8 # Drop
        elif date.year >= 2023:
             trend = 0.1 # Recovery
        
        # Random shock
        shock = np.random.normal(0, 0.5) 
        price = price + trend + shock
        
        # Clip to realistic bounds
        price = max(8.0, min(25.0, price))
        arabica_prices.append(round(price, 2))

    # --- Robusta Price Generation (Lower value, steadier) ---
    # Base price around $5
    robusta_prices = []
    price = 5.0
    for i, date in enumerate(dates):
        trend = 0.02 # Slow inflation
        shock = np.random.normal(0, 0.2)
        price = price + trend + shock
        price = max(2.5, min(9.0, price))
        robusta_prices.append(round(price, 2))

    # --- Demand Generation (Seasonality + Growth) ---
    # Export demand in Metric Tonnes (MT)
    # Arabica: smaller volume, growing fast
    arabica_demand = []
    base_demand = 10.0 # Starts low
    for i, date in enumerate(dates):
        # Seasonality: Harvest peak around Dec-Mar
        month = date.month
        seasonality = 5.0 if month in [12, 1, 2, 3] else 0.0
        
        # Yearly growth
        growth = (date.year - 2015) * 2.0
        
        noise = np.random.normal(0, 2.0)
        demand = base_demand + growth + seasonality + noise
        demand = max(5.0, demand) # Minimum volume
        arabica_demand.append(round(demand, 2))

    # Robusta: larger volume, steady
    robusta_demand = []
    base_demand = 50.0 
    for i, date in enumerate(dates):
        seasonality = 10.0 if date.month in [12, 1, 2] else 0.0
        growth = (date.year - 2015) * 1.5
        noise = np.random.normal(0, 5.0)
        demand = base_demand + growth + seasonality + noise
        demand = max(20.0, demand)
        robusta_demand.append(round(demand, 2))

    # --- Create DataFrame ---
    df = pd.DataFrame({
        'Date': dates,
        'Year': dates.year,
        'Month': dates.month,
        'Arabica_Price_USD_kg': arabica_prices,
        'Robusta_Price_USD_kg': robusta_prices,
        'Arabica_Demand_MT': arabica_demand,
        'Robusta_Demand_MT': robusta_demand
    })
    
    # Save
    output_path = os.path.join(os.getcwd(), 'sri_lanka_coffee_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Synthetic data saved to: {output_path}")
    print(df.head())
    print(df.tail())

if __name__ == "__main__":
    generate_synthetic_data()
