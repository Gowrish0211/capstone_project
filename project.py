import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# ----------------------------
# 1. Load and preprocess data
# ----------------------------
file_path = "dataset.csv"  # update with your path

def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Combine date and time
    df['Datetime'] = pd.to_datetime(
        df['LastUpdatedDate'] + ' ' + df['LastUpdatedTime'],
        format="%d-%m-%Y %H:%M:%S"
    )
    # Sort
    return df.sort_values(['SystemCodeNumber', 'Datetime']).reset_index(drop=True)

# ----------------------------
# 2. Model 1: Baseline Linear
# ----------------------------
BASE_PRICE = 10.0
ALPHA = 2.0

def model1_pricing(df: pd.DataFrame) -> pd.Series:
    def compute_prices(group):
        prices = [BASE_PRICE]
        occ = group['Occupancy'].values
        cap = group['Capacity'].values
        for i in range(1, len(group)):
            delta = ALPHA * (occ[i] / cap[i])
            prices.append(prices[-1] + delta)
        return pd.Series(prices, index=group.index)

    return df.groupby('SystemCodeNumber', group_keys=False).apply(compute_prices)

# ----------------------------
# 3. Model 2: Demand-Based
# ----------------------------
VEHICLE_WEIGHTS = {'car': 1.0, 'bike': 0.5, 'truck': 1.5, 'bus': 2.0}
TRAFFIC_MAP = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
A, B, C, D, E = 1.0, 0.1, 0.05, 0.5, 1.0
LAMBDA = 0.5


def model2_pricing(df: pd.DataFrame) -> pd.Series:
    occ_rate = df['Occupancy'] / df['Capacity']
    veh_w = df['VehicleType'].map(VEHICLE_WEIGHTS)
    traf = df['TrafficConditionNearby'].map(TRAFFIC_MAP)

    raw_demand = (
        A * occ_rate +
        B * df['QueueLength'] -
        C * traf +
        D * df['IsSpecialDay'] +
        E * veh_w
    )
    # normalize
    norm = (raw_demand - raw_demand.min()) / (raw_demand.max() - raw_demand.min())

    price = BASE_PRICE * (1 + LAMBDA * norm)
    return price.clip(lower=0.5*BASE_PRICE, upper=2.0*BASE_PRICE)

# ----------------------------
# 4. Model 3: Competitive
# ----------------------------
def haversine(lat1, lon1, lat2, lon2):
    # radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def build_neighbors(df: pd.DataFrame, radius_km=1.0):
    coords = df[['SystemCodeNumber', 'Latitude', 'Longitude']].drop_duplicates().set_index('SystemCodeNumber')
    neighbors = {}
    for i, (lat1, lon1) in coords.iterrows():
        close = []
        for j, (lat2, lon2) in coords.iterrows():
            if i != j and haversine(lat1, lon1, lat2, lon2) <= radius_km:
                close.append(j)
        neighbors[i] = close
    return neighbors


def model3_pricing(df: pd.DataFrame, price_model2: pd.Series, neighbors: dict) -> pd.Series:
    df = df.copy()
    df['Price_Model2'] = price_model2
    # map neighbor list
    comp_avg = []
    for idx, row in df.iterrows():
        neigh = neighbors[row['SystemCodeNumber']]
        prices = df.loc[(df['Datetime'] == row['Datetime']) & df['SystemCodeNumber'].isin(neigh), 'Price_Model2']
        comp_avg.append(prices.mean() if not prices.empty else row['Price_Model2'])
    df['Comp_Avg'] = comp_avg

    GAMMA = 0.3
    price3 = df['Price_Model2'] * (1 + GAMMA * (df['Price_Model2'] - df['Comp_Avg']) / df['Comp_Avg'])
    # full-lot adjustment
    full = df['Occupancy'] >= df['Capacity']
    price3[full] = np.minimum(price3[full], df.loc[full, 'Comp_Avg'])

    return price3.clip(lower=0.5*BASE_PRICE, upper=2.0*BASE_PRICE)

# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    df = load_and_preprocess(file_path)
    df['Price_Model1'] = model1_pricing(df)
    df['Price_Model2'] = model2_pricing(df)
    neighbors = build_neighbors(df)
    df['Price_Model3'] = model3_pricing(df, df['Price_Model2'], neighbors)

    # Save results
    df.to_csv("dynamic_pricing_output.csv", index=False)
    print("Pricing models computed and saved to dynamic_pricing_output.csv")
