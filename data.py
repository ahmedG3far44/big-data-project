# imports
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import logging


# Generating Weather Synthetic Dataset:
def generate_weather_data(n_rows: int = 5000, city: str = "London"):
    logging.info(f"Starting data generation for {n_rows} rows...")
    data = {}

    # --- 2. Generate Base Dates (Used for dependency generation) ---
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 12, 31)
    time_range_days = (end_date - start_date).days

    # Generate random days within the time range for non-uniform distribution
    random_days = np.random.randint(0, time_range_days, n_rows)
    base_dates = np.array(
        [
            start_date
            + timedelta(
                days=int(d),
                hours=np.random.randint(0, 23),
                minutes=np.random.randint(0, 59),
            )
            for d in random_days
        ]
    )

    # --- 4. Get Clean Seasons (Used for Temperature dependency) ---
    def get_clean_season(date: datetime) -> str:
        """Determines season based on a simple month rule."""
        if 3 <= date.month <= 5:
            return "Spring"
        if 6 <= date.month <= 8:
            return "Summer"
        if 9 <= date.month <= 11:
            return "Autumn"
        return "Winter"

    seasons_clean = np.array([get_clean_season(d) for d in base_dates])

    # --- 5. temperature_c (Float) - Dependency on Season ---
    temperatures = np.zeros(n_rows)
    for s in ["Winter", "Spring", "Summer", "Autumn"]:
        indices = np.where(seasons_clean == s)[0]

        if s == "Winter":  # Range -5 to 15
            temps = np.random.uniform(-5, 15, len(indices))
        elif s == "Spring":  # Range 5 to 20
            temps = np.random.uniform(5, 20, len(indices))
        elif s == "Summer":  # Range 15 to 35
            temps = np.random.uniform(15, 35, len(indices))
        else:  # Autumn, Range 0 to 25
            temps = np.random.uniform(0, 25, len(indices))

        temperatures[indices] = temps

    # Introduce Outliers (-30, 60)
    outlier_count = int(n_rows * 0.005)
    temperatures[np.random.choice(n_rows, outlier_count, replace=False)] = (
        np.random.choice([-30.0, 60.0], outlier_count)
    )

    # Introduce NULLs
    temperatures[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
    data["temperature_c"] = temperatures.round(2)

    # --- 6. humidity (Integer %) ---
    humidity = np.random.randint(20, 101, n_rows)
    # Introduce Outliers (-10, 150)
    outlier_count = int(n_rows * 0.005)
    humidity_outliers = np.random.choice([-10, 150], outlier_count)
    humidity_indices = np.random.choice(n_rows, outlier_count, replace=False)
    humidity[humidity_indices] = humidity_outliers

    # Introduce NULLs
    humidity_nulls = np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
    humidity = humidity.astype(float)  # Convert to float to hold NaN before final cast
    data["humidity"] = humidity

    # --- 7. rain_mm (Float) ---
    rain = np.random.rand(n_rows) * 50  # Typical 0-50mm
    # Introduce Extreme values (120+)
    extreme_count = int(n_rows * 0.002)
    rain[np.random.choice(n_rows, extreme_count, replace=False)] = np.random.uniform(
        120, 150, extreme_count
    )
    # Introduce NULLs
    rain[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
    data["rain_mm"] = rain.round(2)

    # --- 8. wind_speed_kmh (Float) ---
    wind_speed = np.random.rand(n_rows) * 80  # Typical 0-80 km/h
    # Introduce Outliers (200+);
    outlier_count = int(n_rows * 0.002)
    wind_speed[np.random.choice(n_rows, outlier_count, replace=False)] = (
        np.random.uniform(200, 250, outlier_count)
    )
    # Introduce NULLs
    wind_speed[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
    data["wind_speed_kmh"] = wind_speed.round(2)

    # --- 9. visibility_m (Integer/Messy) ---
    visibility_base = np.random.randint(50, 10001, n_rows)
    visibility_messy = visibility_base.astype(object)

    # Extreme values (50,000)
    visibility_messy[np.random.choice(n_rows, int(n_rows * 0.001), replace=False)] = (
        50000
    )
    # NULLs
    visibility_messy[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = (
        np.nan
    )
    # Non-numeric strings (small fraction)
    non_numeric_indices = np.random.choice(n_rows, int(n_rows * 0.01), replace=False)
    visibility_messy[non_numeric_indices] = np.random.choice(
        ["low", "unknown", "N/A", "high"], len(non_numeric_indices)
    )

    data["visibility_m"] = visibility_messy

    # --- 10. weather_condition (Category) ---
    conditions = ["Clear", "Rain", "Fog", "Storm", "Snow"]
    weather_condition = np.random.choice(
        conditions, n_rows, p=[0.4, 0.2, 0.15, 0.05, 0.2]
    )
    # Introduce NULLs
    weather_condition[np.random.choice(n_rows, int(n_rows * 0.03), replace=False)] = (
        np.nan
    )
    # Remove .astype('category') here, will apply after DataFrame creation
    data["weather_condition"] = weather_condition

    # --- 11. air_pressure_hpa (Float) ---
    air_pressure = np.random.uniform(950, 1050, n_rows)
    # Introduce NULLs
    air_pressure[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
    data["air_pressure_hpa"] = air_pressure.round(2)

    # --- 3. city (String) ---
    city_data = np.full(n_rows, city, dtype=object)
    # Introduce NULLs/unknown
    city_data[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
    city_data[np.random.choice(n_rows, int(n_rows * 0.01), replace=False)] = "unknown"
    data["city"] = city_data

    # --- Create DataFrame and Dtype Casts ---
    df = pd.DataFrame(data)

    # Now, convert 'weather_condition' to category type in the DataFrame
    df["weather_condition"] = df["weather_condition"].astype("category")

    # --- 2. date_time (Messy Scenarios) ---
    date_times_messy = base_dates.astype(str)

    # Apply format variations
    indices_v1 = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)  # 10%
    date_times_messy[indices_v1] = [
        d.strftime("%d/%m/%Y %H:%M") for d in base_dates[indices_v1]
    ]

    indices_v2 = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)  # 10%
    date_times_messy[indices_v2] = [
        d.strftime("%Y-%m-%dT%H:%M") for d in base_dates[indices_v2]
    ]

    # Invalid / Garbage
    date_times_messy[np.random.choice(n_rows, int(n_rows * 0.01), replace=False)] = (
        "25/:61"
    )
    date_times_messy[np.random.choice(n_rows, int(n_rows * 0.01), replace=False)] = (
        "2099-13-40"
    )

    # NULLs/unknown
    date_times_messy[np.random.choice(n_rows, int(n_rows * 0.01), replace=False)] = (
        np.nan
    )
    date_times_messy[np.random.choice(n_rows, int(n_rows * 0.01), replace=False)] = (
        "unknown"
    )

    df.insert(1, "date_time", date_times_messy)

    # --- 4. season (Messy Scenarios) ---
    # Season derivation is based on base_dates, now introduce additional NULLs
    seasons_messy = seasons_clean.astype(object)
    seasons_messy[np.random.choice(n_rows, int(n_rows * 0.03), replace=False)] = np.nan
    df.insert(4, "season", seasons_messy)  # Insert as object first
    df["season"] = df["season"].astype("category")  # Then convert the DataFrame column

    # --- 1. weather_id (Integer) ---
    base_ids = np.arange(5001, 5001 + n_rows)
    weather_ids = base_ids.astype(float)

    # Duplicates (10% of rows reuse an ID from the first 1000)
    duplicate_indices = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
    weather_ids[duplicate_indices] = np.random.choice(
        base_ids[:1000], len(duplicate_indices)
    )

    # NULLs (missing ID)
    weather_ids[np.random.choice(n_rows, int(n_rows * 0.005), replace=False)] = np.nan

    # Int64 allows NA (pandas integer type for nullable columns)
    df.insert(0, "weather_id", pd.Series(weather_ids).astype(pd.Int64Dtype()))

    # --- Final Dtype Casting for correct NA handling ---
    df["humidity"] = df["humidity"].astype(pd.Int64Dtype())
    # visibility_m is left as object/string due to non-numeric strings

    logging.info("Data generation complete.")
    print(df.head())
    return df


# Generating Traffic Synthetic Dataset:
def generate_traffic_data(n_rows: int = 5000, city: str = "London"):
    logging.info(f"Starting traffic data generation for {n_rows} rows...")
    data = {}

    # --- 1. Base Dates (Used for dependency generation) ---
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    time_range_seconds = int((end_date - start_date).total_seconds())

    # Generate random seconds within the time range
    random_seconds = np.random.randint(0, time_range_seconds, n_rows)
    # Ensure times are rounded to the nearest minute/hour for traffic measurement consistency
    base_dates = np.array(
        [
            start_date + timedelta(seconds=int(s))
            for s in random_seconds  # Cast s to int
        ]
    )

    # --- 4. area (String) ---
    # Used as a dependency for vehicle count/speed (e.g., Chelsea might be slower than Southwark)
    areas = ["Camden", "Chelsea", "Islington", "Southwark", "Kensington", "Westminster"]
    area_weights = [
        0.15,
        0.20,
        0.15,
        0.15,
        0.25,
        0.10,
    ]  # Weights to simulate varying sample rates per area

    clean_areas = np.random.choice(areas, n_rows, p=area_weights)
    area_data = clean_areas.astype(object)

    # Introduce NULL area values
    area_data[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
    data["area"] = area_data

    # --- 5. vehicle_count (Integer) - Dependency on Area ---
    vehicle_counts = np.zeros(n_rows)
    for area in areas:
        indices = np.where(clean_areas == area)[0]

        # Chelsea/Westminster (Central/Dense) tend to have higher vehicle counts
        if area in ["Chelsea", "Westminster", "Kensington"]:
            counts = np.random.randint(100, 5001, len(indices))
        else:  # Other areas
            counts = np.random.randint(0, 3001, len(indices))

        vehicle_counts[indices] = counts

    # Introduce Outliers (20,000+)
    outlier_count = int(n_rows * 0.003)
    vehicle_counts[np.random.choice(n_rows, outlier_count, replace=False)] = (
        np.random.randint(20000, 30000, outlier_count)
    )

    # Introduce NULLs
    vehicle_counts_nulls = np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
    # Convert to float temporarily to hold NaN
    vehicle_counts = vehicle_counts.astype(float)
    vehicle_counts[vehicle_counts_nulls] = np.nan
    data["vehicle_count"] = vehicle_counts

    # --- 6. avg_speed_kmh (Float) - Dependency on Vehicle Count (Negative correlation) ---
    # Base speed for low traffic: 60-120 km/h (highway context)
    # Base speed for high traffic: 3-50 km/h (city context)
    avg_speeds = np.random.uniform(3, 120, n_rows)

    # Apply negative correlation: High count = Low speed
    high_traffic_indices = np.where(vehicle_counts > 3000)[0]
    avg_speeds[high_traffic_indices] = np.random.uniform(
        3, 30, len(high_traffic_indices)
    )

    # Apply positive correlation: Low count = High speed
    low_traffic_indices = np.where(vehicle_counts < 500)[0]
    avg_speeds[low_traffic_indices] = np.random.uniform(
        40, 100, len(low_traffic_indices)
    )

    # Introduce Invalid values (negative speeds)
    invalid_count = int(n_rows * 0.005)
    avg_speeds[np.random.choice(n_rows, invalid_count, replace=False)] = (
        np.random.uniform(-10.0, -1.0, invalid_count)
    )

    # Introduce NULLs
    avg_speeds[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
    data["avg_speed_kmh"] = avg_speeds.round(2)

    # --- 7. accident_count (Integer) - Dependency on Congestion and Speed ---
    accident_counts = np.random.poisson(lam=1, size=n_rows)  # Most are 0 or 1
    accident_counts[accident_counts > 10] = 10  # Cap at 10 for 'expected' range

    # High Congestion/Low Speed -> Higher accident risk (set to 5 or less)
    high_risk_indices = np.where((vehicle_counts > 4000) | (avg_speeds < 10))[0]
    accident_counts[high_risk_indices] = np.random.poisson(
        lam=2, size=len(high_risk_indices)
    )

    # Introduce Extreme values (50+)
    extreme_count = int(n_rows * 0.001)
    accident_counts[np.random.choice(n_rows, extreme_count, replace=False)] = (
        np.random.randint(50, 70, extreme_count)
    )

    # Introduce NULLs
    accident_counts_nulls = np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
    accident_counts = accident_counts.astype(float)
    accident_counts[accident_counts_nulls] = np.nan
    data["accident_count"] = accident_counts

    # --- 8. congestion_level (Category) - Dependency on Vehicle Count/Speed ---
    congestion = np.full(n_rows, "Medium", dtype=object)

    # High Congestion (High count, Low speed)
    high_indices = np.where((vehicle_counts > 4500) | (avg_speeds < 15))[0]
    congestion[high_indices] = "High"

    # Low Congestion (Low count, High speed)
    low_indices = np.where((vehicle_counts < 1000) & (avg_speeds > 60))[0]
    congestion[low_indices] = "Low"

    # Introduce NULLs
    congestion[np.random.choice(n_rows, int(n_rows * 0.03), replace=False)] = np.nan
    data["congestion_level"] = congestion  # Assign numpy array directly

    # --- 9. road_condition (Category) ---
    conditions = ["Dry", "Wet", "Snowy", "Damaged"]
    # Wet/Dry are most common
    road_condition = np.random.choice(conditions, n_rows, p=[0.7, 0.2, 0.05, 0.05])

    # Introduce NULLs
    road_condition[np.random.choice(n_rows, int(n_rows * 0.03), replace=False)] = np.nan
    data["road_condition"] = road_condition  # Assign numpy array directly

    # --- 10. visibility_m (Integer) ---
    visibility = np.random.randint(50, 10001, n_rows)  # Typical: 50-10,000

    # Introduce NULLs
    visibility_nulls = np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
    visibility = visibility.astype(float)
    visibility[visibility_nulls] = np.nan
    data["visibility_m"] = visibility

    # --- 3. city (String) ---
    city_data = np.full(n_rows, city, dtype=object)
    # Introduce NULLs
    city_data[np.random.choice(n_rows, int(n_rows * 0.01), replace=False)] = np.nan
    data["city"] = city_data

    # --- Create DataFrame and Dtype Casts ---
    df = pd.DataFrame(data)

    # Convert to category Dtype after DataFrame creation
    df["congestion_level"] = df["congestion_level"].astype("category")
    df["road_condition"] = df["road_condition"].astype("category")

    # --- 2. date_time (Messy Scenarios) ---
    date_times_messy = base_dates.astype(str)

    # Apply format variations (10% of rows for each major variation)
    indices_v1 = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
    date_times_messy[indices_v1] = [
        d.strftime("%d/%m/%Y %H:%M") for d in base_dates[indices_v1]
    ]

    indices_v2 = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
    date_times_messy[indices_v2] = [
        d.strftime("%Y-%m-%dT%H:%M:00") for d in base_dates[indices_v2]
    ]  # Extra seconds zeroed

    # Invalid / Garbage
    date_times_messy[np.random.choice(n_rows, int(n_rows * 0.005), replace=False)] = (
        "TBD"
    )
    date_times_messy[np.random.choice(n_rows, int(n_rows * 0.005), replace=False)] = (
        "2099-00-00 99:99"
    )

    # NULLs
    date_times_messy[np.random.choice(n_rows, int(n_rows * 0.01), replace=False)] = (
        np.nan
    )

    df.insert(1, "date_time", date_times_messy)

    # --- 1. traffic_id (Integer) ---
    base_ids = np.arange(9001, 9001 + n_rows)
    traffic_ids = base_ids.astype(float)

    # Duplicates (10% of rows reuse an ID from the first 1000)
    duplicate_indices = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
    traffic_ids[duplicate_indices] = np.random.choice(
        base_ids[:1000], len(duplicate_indices)
    )

    # NULLs (missing ID)
    traffic_ids[np.random.choice(n_rows, int(n_rows * 0.005), replace=False)] = np.nan

    # Int64 allows NA (pandas integer type for nullable columns)
    df.insert(0, "traffic_id", traffic_ids)

    # --- Final Dtype Casting for correct NA handling ---
    df["traffic_id"] = df["traffic_id"].astype("Int64")
    df["vehicle_count"] = df["vehicle_count"].astype("Int64")
    df["accident_count"] = df["accident_count"].astype("Int64")
    df["visibility_m"] = df["visibility_m"].astype("Int64")

    logging.info("Data generation complete.")
    print(df.head())
    return df

generate_weather_data(n_rows=2000)
generate_traffic_data(n_rows=2000)
