import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import difflib

data1=pd.read_excel('2024-01-29-12-40-PM-1000.xlsx')
data2 = pd.read_excel('PT_25_Sensor_Data_1706771237585.xlsx')

# Loop through every column in the DataFrame
for col_name in data1.columns:
    
    # Check if the column contains any missing values
    if data1[col_name].isnull().sum() > 0:
        if data1[col_name].dtype in ['float64', 'int64']:  # Numeric columns (float and int)
            data1[col_name].fillna(data1[col_name].mean(), inplace=True)  # Fill missing values with mean
            print(f"Filled missing values in numeric column '{col_name}' with the mean.")
        elif data1[col_name].dtype == 'object':  # Non-numeric columns (object type)
            data1[col_name].fillna(data1[col_name].mode()[0], inplace=True)  # Fill missing values with mode
            print(f"Filled missing values in non-numeric column '{col_name}' with the mode.")
    else:
        print(f"No missing values in column '{col_name}'.")

# Remove the prefix and keep the actual column name
# Save changes directly to the existing Excel file (overwrite)
data1.columns = data1.columns.str.split('::').str[-1]  # Rename the columns
data1.to_excel('ecuData.xlsx', index=False)  # Overwrite the existing file
# Replace common placeholder values (like empty strings and spaces) with NaN in `df1`
data1.replace(["", " ", 0], pd.NA, inplace=True)

# Predefined mapping
predefined_mapping = {
    'current[a]': 'current',
    'soc[%]': 'soc',
    'full_capacity[ah]': 'res capacity',
    'vcell1_voltage[v]': 'cellvoltage1',
    'vcell2_voltage[v]': 'cellvoltage2',
    'vcell3_voltage[v]': 'cellvoltage3',
    'vcell4_voltage[v]': 'cellvoltage4',
    'vcell5_voltage[v]': 'cellvoltage5',
    'vcell6_voltage[v]': 'cellvoltage6',
    'vcell7_voltage[v]': 'cellvoltage7',
    'vcell8_voltage[v]': 'cellvoltage8',
    'vcell9_voltage[v]': 'cellvoltage9',
    'vcell10_voltage[v]': 'cellvoltage10',
    'vcell11_voltage[v]': 'cellvoltage11',
    'vcell12_voltage[v]': 'cellvoltage12',
    'vcell13_voltage[v]': 'cellvoltage13',
    'vcell14_voltage[v]': 'cellvoltage14',
    'total_voltage[v]': 'battery pack voltage',
    'vmin[v]': 'mincellvoltage',
    'vmax[v]': 'maxcellvoltage',
    'vmin_number': 'mincellid',
    'vmax_number': 'maxcellid',
    'cycles_times[count]': 'charge cycle',
    'average_temperature[*c]': 'temperature',
    'maximum_temperature[*c]': 'maxcelltemp',
    'minimum_temperature[*c]': 'mincelltemp',
    't1[*c]': 'temp1',
    't2[*c]': 'temp2',
    't3[*c]': 'temp3',
    't4[*c]': 'temp4',
    't5[*c]': 'temp5',
    't6[*c]': 'temp6',
    'dte': 'range available'
}

# Load datasets
file1_path = "2024-01-29-12-40-PM-1000.xlsx"
file2_path = "PT_25_Sensor_Data_1706771237585.xlsx"
data1 = pd.read_excel(file1_path)
data2 = pd.read_excel(file2_path)

# Standardize column names (strip spaces and convert to lowercase)
data1.columns = data1.columns.str.strip().str.lower()
data2.columns = data2.columns.str.strip().str.lower()

# Validate columns against predefined mapping
missing_columns_data1 = [col for col in predefined_mapping.keys() if col not in data1.columns]
missing_columns_data2 = [col for col in predefined_mapping.values() if col not in data2.columns]

if missing_columns_data1:
    raise KeyError(f"The following columns are missing in Dataset 1: {missing_columns_data1}")
if missing_columns_data2:
    raise KeyError(f"The following columns are missing in Dataset 2: {missing_columns_data2}")

# Filter Dataset 1 to include only mapped columns
data1_filtered = data1[['time[s]'] + list(predefined_mapping.keys())].dropna()

# Ensure TimestampIST in Dataset 2 is datetime
data2["timestampist"] = pd.to_datetime(data2["timestampist"], errors="coerce")

# Start timestamp for Dataset 1
start_time = pd.Timestamp("2024-01-29 12:40:00")

# Function to aggregate Dataset 1
def aggregate_60s_interval(df, start_time):
    df["absolute_time"] = start_time + pd.to_timedelta(df["time[s]"], unit="s")
    df["time_group"] = df["absolute_time"].dt.floor("60s")
    aggregated_mean = df.groupby("time_group").mean().reset_index()
    aggregated_median = df.groupby("time_group").median().reset_index()
    return aggregated_mean, aggregated_median

# Filter Dataset 2 to match time range of Dataset 1
time_max = start_time + pd.Timedelta(data1_filtered["time[s]"].max(), unit="s")
data2_filtered = data2[
    (data2["timestampist"] >= start_time) & (data2["timestampist"] <= time_max)
]

# Debugging: Check filtered Dataset 2 range
print("Dataset 2 filtered TimestampIST range:", data2_filtered["timestampist"].min(), "-", data2_filtered["timestampist"].max())

# Process predefined mappings
formatted_results = []

for col1, col2 in predefined_mapping.items():
    print(f"Processing column: {col1} (Dataset 1) -> {col2} (Dataset 2)")
    
    # Aggregate Dataset 1 for the current column
    data1_subset = data1_filtered[["time[s]", col1]].dropna()
    data1_aggregated_mean, data1_aggregated_median = aggregate_60s_interval(data1_subset, start_time)
    
    # Merge with Dataset 2
    comparison_mean = pd.merge(
        data1_aggregated_mean,
        data2_filtered[["timestampist", col2]],
        left_on="time_group",
        right_on="timestampist",
        how="inner",
    )
    comparison_median = pd.merge(
        data1_aggregated_median,
        data2_filtered[["timestampist", col2]],
        left_on="time_group",
        right_on="timestampist",
        how="inner",
    )
    
    # Combine results for both mean and median
    if not comparison_mean.empty and not comparison_median.empty:
        for idx, row_mean in comparison_mean.iterrows():
            # Find corresponding median row for the same timestamp
            row_median = comparison_median.loc[comparison_median["time_group"] == row_mean["time_group"]].iloc[0]
            
            formatted_results.append({
                "Timestamp": row_mean["time_group"],
                "Data Column": col1,
                "Mean": row_mean[col1],
                "Mean Deviation": row_mean[col1] - row_mean[col2],
                "Median": row_median[col1],
                "Median Deviation": row_median[col1] - row_median[col2],
            })
    
    # Add a blank row to separate data columns
    formatted_results.append({
        "Timestamp": None,
        "Data Column": None,
        "Mean": None,
        "Mean Deviation": None,
        "Median": None,
        "Median Deviation": None,
    })

# Display results in the desired format
if formatted_results:
    results_df = pd.DataFrame(formatted_results)
    print(results_df)
    
    # Optionally save to Excel
    results_df.to_excel("results.xlsx", index=False)
    print("Results saved to 'results.xlsx'")
else:
    print("No data to display. Check your datasets and mapping.")

# Load the formatted results
results_file = "results.xlsx"
results_df = pd.read_excel(results_file)

# Remove blank rows for plotting
results_cleaned = results_df.dropna()

# Ensure timestamp is in datetime format
results_cleaned["Timestamp"] = pd.to_datetime(results_cleaned["Timestamp"])

# Plot deviations for mean and median
plt.figure(figsize=(16, 8))
sns.set_theme(style="whitegrid")

# Bar plot of deviations
sns.barplot(
    data=results_cleaned.melt(
        id_vars=["Timestamp", "Data Column"],
        value_vars=["Mean Deviation", "Median Deviation"],
        var_name="Deviation Type",
        value_name="Deviation"
    ),
    x="Data Column",
    y="Deviation",
    hue="Deviation Type",
    errorbar=None
)

plt.title("Deviations of Mean and Median", fontsize=16)
plt.xlabel("Data Column", fontsize=12)
plt.ylabel("Deviation", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.legend(title="Deviation Type")
plt.tight_layout()
plt.savefig("deviation_barplot.png")  # Save the plot as an image
plt.show()

# Line plot over timestamps
plt.figure(figsize=(16, 8))

for col in results_cleaned["Data Column"].unique():
    subset = results_cleaned[results_cleaned["Data Column"] == col]
    plt.plot(subset["Timestamp"], subset["Mean Deviation"], label=f"{col} - Mean", marker='o')
    plt.plot(subset["Timestamp"], subset["Median Deviation"], label=f"{col} - Median", marker='x')

plt.title("Deviations Over Time", fontsize=16)
plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("Deviation", fontsize=12)
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.savefig("deviation_lineplot.png")  # Save the plot as an image
plt.show()

# Load the cleaned results
results_file = "results.xlsx"
results_df = pd.read_excel(results_file)

# Remove blank rows for analysis
results_cleaned = results_df.dropna()

# Add absolute deviations for mean and median
results_cleaned["Absolute Mean Deviation"] = results_cleaned["Mean Deviation"].abs()
results_cleaned["Absolute Median Deviation"] = results_cleaned["Median Deviation"].abs()

# Calculate overall statistics
mean_deviation_summary = results_cleaned["Absolute Mean Deviation"].mean()
median_deviation_summary = results_cleaned["Absolute Median Deviation"].mean()

print("Average Absolute Deviations:")
print(f"Mean: {mean_deviation_summary}")
print(f"Median: {median_deviation_summary}")

# Determine which is better
if mean_deviation_summary < median_deviation_summary:
    print("Conclusion: Mean is the better aggregation method for this data.")
else:
    print("Conclusion: Median is the better aggregation method for this data.")

# Group by data column for detailed comparison
column_comparison = results_cleaned.groupby("Data Column")[["Absolute Mean Deviation", "Absolute Median Deviation"]].mean()
print("\nAverage Absolute Deviations by Data Column:")
print(column_comparison)

# Determine the preferred method per column
column_comparison["Preferred Method"] = column_comparison.apply(
    lambda row: "Mean" if row["Absolute Mean Deviation"] < row["Absolute Median Deviation"] else "Median",
    axis=1
)
print("\nPreferred Aggregation Method by Data Column:")
print(column_comparison)
