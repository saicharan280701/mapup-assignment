from typing import Dict, List

import pandas as pd

## Question 1: Reverse List by N Elements

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    for i in range(0, len(lst), n):
        end_index = min(i + n, len(lst))
        lst[i:end_index] = lst[i:end_index][::-1]
    return lst

user_input = input("Input: ")
n = int(input("n: "))
input_list = list(map(int, user_input.strip('[]').split(',')))
output = reverse_by_n_elements(input_list, n)
print("Output:", output)


## Question 2: Lists & Dictionaries


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here.
    length_dict = {}
    for s in strings:
        length_dict.setdefault(len(s), []).append(s)  
    return dict(sorted(length_dict.items()))

user_input = input("Input (comma-separated strings): ")
input_list = [s.strip() for s in user_input.split(',')]  
output = group_by_length(input_list)
print("Output:", output)

## Question 3: Flatten a Nested Dictionary

from typing import Dict, Any
import ast

def flatten_dictionary(nested_dict: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    items = {}
    for key, value in nested_dict.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        
        if isinstance(value, dict):
            items.update(flatten_dictionary(value, new_key, sep=sep))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    items.update(flatten_dictionary(item, f"{new_key}[{index}]", sep=sep))
                else:
                    items[f"{new_key}[{index}]"] = item
        else:
            items[new_key] = value
            
    return items

user_input = input("Input: ")

try:
    nested_dict = ast.literal_eval(user_input)
    if not isinstance(nested_dict, dict):
        raise ValueError("Input must be a dictionary.")
except Exception as e:
    print(f"Invalid input: {e}")
else:

    flattened_dict = flatten_dictionary(nested_dict)
    print("Output:")
    print(flattened_dict)
    
##Question 4: Generate Unique Permutations

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
     def backtrack(start: int):
        if start == len(nums):
            result.append(nums[:])
            return
        
        seen = set()  
        for i in range(start, len(nums)):
            if nums[i] in seen:
                continue  
            seen.add(nums[i])
            nums[start], nums[i] = nums[i], nums[start] 
            backtrack(start + 1)  # Recurse

    result = []
    nums.sort()  
    backtrack(0)
    return result

input_list = [1, 1, 2]
output = unique_permutations(input_list)
print("Output:", output)

## Question 5: Find All Dates in a Text

import re
from typing import List

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    pattern = r'\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b'
    
    matches = re.findall(pattern, text)
    
    return matches

text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print("Output:", output)

## Question 6: Decode Polyline, Convert to DataFrame with Distances

    

import pandas as pd
import polyline
import numpy as np

def haversine(lat1, lon1, lat2, lon2):

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371000  # Radius of Earth in meters
    return c * r

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:

    coordinates = polyline.decode(polyline_str)

    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])

    distances = [0]  # First point has distance 0
    for i in range(1, len(df)):
        dist = haversine(df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'],
                         df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distances.append(dist)

    df['distance'] = distances

    return df

polyline_input = input("Input: ")
output_df = polyline_to_dataframe(polyline_input)
print("Output DataFrame:")
print(output_df)

## Question 7: Matrix Rotation and Transformation

def rotate_and_transform_matrix(matrix: List[List[int]]) -> List[List[int]]:

    n = len(matrix)
    
    rotated_matrix = [[0] * n for _ in range(n)]  # Initialize a new matrix for rotation
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]

    print(f"rotated_matrix = {rotated_matrix}")

    transformed_matrix = [[0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            transformed_matrix[i][j] = row_sum + col_sum

    print(f"final_matrix = {transformed_matrix}")

    return transformed_matrix

def main():
    user_input = input("Input: ")
    
    matrix = ast.literal_eval(user_input)

    output_matrix = rotate_and_transform_matrix(matrix)

if __name__ == "__main__":
    main()
    
## Question 8: Time Check


import pandas as pd
from datetime import datetime, timedelta
file_path = r'C:\Users\chara\Downloads\dataset-1.csv'

df = pd.read_csv(file_path)

day_to_offset = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}

today = datetime.now()

def get_full_datetime(day, time):
    """Convert a given day and time to a full datetime object."""
    day_offset = day_to_offset[day]
    date_of_week = today - timedelta(days=today.weekday() - day_offset)
    return datetime.strptime(f"{date_of_week.date()} {time}", '%Y-%m-%d %H:%M:%S')

df['start'] = df.apply(lambda row: get_full_datetime(row['startDay'], row['startTime']), axis=1)
df['end'] = df.apply(lambda row: get_full_datetime(row['endDay'], row['endTime']), axis=1)

def time_check(df: pd.DataFrame) -> pd.Series:
    """
    Check if the timestamps for each unique (id, id_2) pair cover a full 24-hour period
    and span all 7 days of the week.

    Args:
        df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
        pd.Series: A boolean series indicating whether each (id, id_2) pair has incorrect timestamps.
    """
    grouped = df.groupby(['id', 'id_2'])

    results = []

    for (id_val, id_2_val), group in grouped:
        start_time = group['start'].min()
        end_time = group['end'].max()
        
        covers_24_hours = (end_time - start_time) >= timedelta(days=1)
        
        days_covered = group['startDay'].unique()
        covers_7_days = len(days_covered) == 7
        
        results.append(((id_val, id_2_val), not (covers_24_hours and covers_7_days)))

    return pd.Series(dict(results))

result = time_check(df)

print(result)
