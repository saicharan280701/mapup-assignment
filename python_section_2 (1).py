
## Question 9: Distance Matrix Calculation

import pandas as pd


import pandas as pd
import numpy as np

def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:

    unique_ids = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))

    distance_matrix = pd.DataFrame(np.nan, index=unique_ids, columns=unique_ids)

    for _, row in df.iterrows():
        distance_matrix.at[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.at[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    np.fill_diagonal(distance_matrix.values, 0)

    for k in unique_ids:
        for i in unique_ids:
            for j in unique_ids:
                if pd.notna(distance_matrix.at[i, k]) and pd.notna(distance_matrix.at[k, j]):
                    new_distance = distance_matrix.at[i, k] + distance_matrix.at[k, j]
                    if pd.isna(distance_matrix.at[i, j]) or new_distance < distance_matrix.at[i, j]:
                        distance_matrix.at[i, j] = new_distance

    return distance_matrix

file_path = r'C:\Users\chara\Downloads\dataset-2.csv'
df = pd.read_csv(file_path)

distance_matrix = calculate_distance_matrix(df)

print(distance_matrix)


## Question 10: Unroll Distance Matrix



import pandas as pd

def unroll_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unrolls a distance matrix into a long-format DataFrame with columns id_start, id_end, and distance.

    Args:
        df (pd.DataFrame): The distance matrix with IDs as both index and columns.

    Returns:
        pd.DataFrame: A DataFrame with columns id_start, id_end, and distance.
    """
    
    stacked = df.stack()
    
    filtered = stacked[stacked.index.get_level_values(0) != stacked.index.get_level_values(1)]
    
    unrolled_df = filtered.reset_index(name='distance')
    
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    
    return unrolled_df

unrolled_df = unroll_distance_matrix(distance_matrix)
print(unrolled_df)

## Question 11: Finding IDs within Percentage Threshold

import pandas as pd

def find_ids_within_ten_percentage_threshold(df: pd.DataFrame, reference_id: int) -> list:
    """
    Finds IDs from the id_start column that are within 10% of the average distance 
    of the given reference_id.

    Args:
        df (pd.DataFrame): DataFrame containing id_start, id_end, and distance columns.
        reference_id (int): The id_start value for which to calculate the average distance.

    Returns:
        list: A sorted list of id_start values within the 10% threshold of the reference_id's average distance.
    """
    distances = df[df['id_start'] == reference_id]['distance']
    
    if distances.empty:
        print(f"No distances found for reference id: {reference_id}")
        return []

    average_distance = distances.mean()
    
    lower_bound = average_distance * 0.90
    upper_bound = average_distance * 1.10
    
    ids_within_threshold = df[(df['distance'] >= lower_bound) & (df['distance'] <= upper_bound)]['id_start']
    
    result = sorted(ids_within_threshold.unique())
    
    return result

if __name__ == "__main__":
    data = {
        'id_start': [1014000, 1014000, 1014002, 1014003, 1030000, 1014000],
        'id_end': [1014001, 1014002, 1014003, 1030000, 1014002, 1014003],
        'distance': [100, 150, 200, 300, 250, 120]
    }
    
    unrolled_df = pd.DataFrame(data)

    reference_id = 1014000
    result_ids = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
    print(f"IDs within 10% of the average distance for {reference_id}: {result_ids}")

    
## Question 12: Calculate Toll Rate

import pandas as pd

def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate toll rates based on vehicle types and add new columns to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'distance' column.

    Returns:
        pd.DataFrame: The original DataFrame with additional columns for toll rates.
    """
    
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    for vehicle, rate in rate_coefficients.items():
        df[vehicle] = df['distance'] * rate

    return df
toll_rates_df = calculate_toll_rate(unrolled_df)
print(toll_rates_df)

## Question 13: Calculate Time-Based Toll Rates

import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pd.DataFrame): Input DataFrame containing toll rates.

    Returns:
        pd.DataFrame: DataFrame with time-based toll rates.
    """
    weekday_discount_factors = {
        'morning': 0.8,   # 00:00 to 10:00
        'afternoon': 1.2, # 10:00 to 18:00
        'evening': 0.8    # 18:00 to 23:59
    }
    weekend_discount_factor = 0.7

    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    new_rows = []

    unique_pairs = df[['id_start', 'id_end', 'moto', 'car', 'rv', 'bus', 'truck', 'distance']].drop_duplicates()

    for _, row in unique_pairs.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']
        
        for day in days_of_week[:5]: 
            for period, time_range in zip(
                ['morning', 'afternoon', 'evening'],
                [(time(0, 0), time(10, 0)), (time(10, 0), time(18, 0)), (time(18, 0), time(23, 59))]
            ):
                new_rows.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance,
                    'start_day': day,
                    'start_time': time_range[0],
                    'end_day': day,
                    'end_time': time_range[1],
                    **{vehicle: row[vehicle] * weekday_discount_factors[period] for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
                })

        for day in days_of_week[5:]: 
            new_rows.append({
                'id_start': id_start,
                'id_end': id_end,
                'distance': distance,
                'start_day': day,
                'start_time': time(0, 0),
                'end_day': day,
                'end_time': time(23, 59),
                **{vehicle: row[vehicle] * weekend_discount_factor for vehicle in ['moto', 'car', 'rv', 'bus', 'truck']}
            })

    return pd.DataFrame(new_rows)

def create_sample_toll_rates_df() -> pd.DataFrame:
    """
    Create a sample DataFrame to simulate the toll rates.
    
    Returns:
        pd.DataFrame: Sample DataFrame with id_start, id_end, distance, and toll rates.
    """
    data = {
        'id_start': [1001400, 1001402, 1001404, 1001408, 1001400, 1001408],
        'id_end': [1001402, 1001404, 1001406, 1001410, 1001402, 1001410],
        'distance': [10.0, 20.0, 30.0, 11.1, 9.7, 11.1],
        'moto': [9.7, 20.2, 16.0, 12.5, 9.7, 12.5],
        'car': [12.0, 22.0, 18.0, 15.0, 12.0, 15.0],
        'rv': [15.0, 25.0, 20.0, 17.0, 15.0, 17.0],
        'bus': [18.0, 28.0, 22.0, 19.0, 18.0, 19.0],
        'truck': [25.0, 35.0, 30.0, 27.0, 25.0, 27.0]
    }
    
    return pd.DataFrame(data)

toll_rates_df = create_sample_toll_rates_df()

result_df = calculate_time_based_toll_rates(toll_rates_df)

print(result_df)

