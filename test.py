import boto3
import duckdb
import pandas as pd
import numpy as np
import io
import os
import random
from botocore.client import Config
import logging
from datetime import datetime, timedelta


# ==============================================================================
# 🎯 FIX 1: Missing Helper Function (clean_date_time)
# The cleaning functions rely on this helper function which was not defined.
# ==============================================================================
def clean_date_time(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Cleans and standardizes the date_time column.
    Handles mixed formats, invalid/garbage entries, and NULLs.
    """
    # 1. Standardize to datetime objects, coercing invalid/garbage entries to NaT
    df[col_name] = pd.to_datetime(
        df[col_name], errors="coerce", infer_datetime_format=True
    )

    # 2. Drop rows where date_time is NaT (invalid/garbage/NULL)
    df.dropna(subset=[col_name], inplace=True)

    # 3. Standardize the format of the column
    df[col_name] = df[col_name].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


class DataLakeManager:
    def __init__(self, endpoint_url, access_key, secret_key):
        self.endpoint = endpoint_url
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

        # DuckDB Setup
        self.con = duckdb.connect()
        self._configure_duckdb(endpoint_url, access_key, secret_key)

        # Ensure buckets exist
        self._ensure_buckets_exist(["bronze", "silver", "gold", "logs"])

        self.log_message("Data Lake Manager Initialized.")

    def _configure_duckdb(self, url, key, secret):
        """Configure DuckDB HTTPFS to read/write from MinIO"""

        # MinIO/DuckDB often needs the endpoint without http/https scheme
        s3_endpoint = url.replace("http://", "").replace("https://", "")

        # DuckDB SET s3_endpoint needs the host and port (e.g., 'localhost:9000')
        # If the port is missing from the input URL, we assume the default MinIO port 9000
        if ":" not in s3_endpoint:
            s3_endpoint += ":9000"

        self.con.execute("INSTALL httpfs; LOAD httpfs;")
        self.con.execute(f"SET s3_endpoint='{s3_endpoint}';")
        self.con.execute(f"SET s3_access_key_id='{key}';")
        self.con.execute(f"SET s3_secret_access_key='{secret}';")
        self.con.execute(
            "SET s3_use_ssl=false;"
        )  # Assumes http (MinIO default in local setups)
        self.con.execute("SET s3_url_style='path';")
        self.con.execute("SET s3_region='us-east-1';")
        self.log_message("DuckDB HTTPFS configured for S3/MinIO.")

    def _ensure_buckets_exist(self, buckets):
        # ... (Bucket creation logic is mostly fine, but simplified the exception handling)
        try:
            existing = [b["Name"] for b in self.s3.list_buckets().get("Buckets", [])]
        except Exception as e:
            self.log_message(f"Error listing buckets: {e}")
            existing = []

        for b in buckets:
            if b not in existing:
                self.log_message(f"Creating bucket: {b}")
                try:
                    # Use a standard S3/MinIO creation call
                    self.s3.create_bucket(Bucket=b)
                    self.log_message(f"Bucket {b} created successfully")
                except Exception as e:
                    # Ignore error if the bucket already exists (common concurrency issue)
                    if "BucketAlreadyOwnedByYou" in str(e):
                        self.log_message(f"Bucket {b} already exists.")
                    else:
                        self.log_message(f"Error creating bucket {b}: {e}")

    def log_message(self, message):
        # ... (log_message logic is mostly fine)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{ts}] {message}"
        print(full_msg)

        fname = f"datalake_log_{datetime.now().strftime('%Y-%m-%d')}.txt"

        try:
            # FIX: Use self.s3 to check for object existence properly
            try:
                obj = self.s3.get_object(Bucket="logs", Key=fname)
                old = obj["Body"].read().decode("utf-8")
            except self.s3.exceptions.NoSuchKey:
                old = ""
            except Exception:
                old = ""  # Catch other potential read errors

            new = old + full_msg + "\n"

            self.s3.put_object(Bucket="logs", Key=fname, Body=new.encode("utf-8"))
        except Exception as e:
            print(f"Failed to write logs: {e}")

    def upload_data(self, data, layer, filename):
        self.log_message(f"Uploading data to {layer}/{filename}...")

        if isinstance(data, pd.DataFrame):
            csv_buffer = io.BytesIO()
            data.to_parquet(csv_buffer, index=False)
            file_content = csv_buffer.getvalue()
            content_type = "application/x-parquet"
            if not filename.endswith(".parquet"):
                filename = filename.replace(".csv", ".parquet")
        else:
            file_content = data.encode("utf-8")
            content_type = "text/csv"

        try:
            self.s3.put_object(
                Bucket=layer,
                Key=filename,
                Body=file_content,
                ContentType=content_type,
            )
            self.log_message(f"✓ Upload successful to {layer}/{filename}")
            return True
        except Exception as e:
            self.log_message(f"✗ Upload failed: {e}")
            return False

    def run_etl_pipeline(self):
        """Complete ETL pipeline"""
        self.log_message("Starting ETL Pipeline...")

        try:
            self.log_message("[1] Creating Silver Layer (Cleaning Data)")

            raw_weather = self.download_data("bronze", "raw_weather.parquet")
            raw_traffic = self.download_data("bronze", "raw_traffic.parquet")

            self._create_silver_layer(raw_weather, raw_traffic)
        except Exception as e:
            self.log_message(f"✗ Error creating silver layer: {e}")
            return False

        # Step 2: Create Gold Layer (Aggregated Data)
        try:
            self.log_message("[2] Creating Gold Layer (Aggregations)")
            self._create_gold_layer()
        except Exception as e:
            self.log_message(f"✗ Error creating gold layer: {e}")
            return False

        self.log_message("✓ ETL Pipeline completed successfully!")
        return True

    def download_data(self, layer, filename):
        """Downloads data from S3/MinIO and returns a DataFrame."""
        self.log_message(f"Downloading data from {layer}/{filename}...")
        try:
            s3_path = f"s3://{layer}/{filename}"
            # Use DuckDB to read the parquet file directly from S3
            df = self.con.execute(f"SELECT * FROM '{s3_path}'").df()
            self.log_message(f"✓ Download successful: {len(df)} rows loaded.")
            return df
        except Exception as e:
            self.log_message(f"✗ Download failed for {layer}/{filename}: {e}")
            raise

    # ==============================================================================
    # 🎯 FIX 2: Convert to Class Methods
    # The cleaning and generation functions must take 'self' as the first argument
    # to be callable within the class (e.g., self.clean_traffic_data()).
    # ==============================================================================

    def clean_traffic_data(self, df: pd.DataFrame):
        df["traffic_id"].fillna(-1, inplace=True)
        df.drop_duplicates(
            subset=df.columns.difference(["traffic_id"]), keep="first", inplace=True
        )
        df = df[df["traffic_id"] != -1].copy()
        df["traffic_id"] = pd.to_numeric(df["traffic_id"], errors="coerce").astype(
            "Int64"
        )

        # FIX: Calling the global helper function
        df = clean_date_time(df, "date_time")

        df["city"].fillna("London", inplace=True)
        df["city"] = df["city"].astype(str).str.strip()
        df = df[df["city"].isin(["London"])].copy()

        standard_areas = ["Camden", "Chelsea", "Islington", "Southwark", "Kensington"]
        df["area"] = df["area"].astype(str).str.strip().str.title()

        mode_area = df["area"].mode()[0] if not df["area"].mode().empty else "Unknown"

        df["area"] = np.where(~df["area"].isin(standard_areas), mode_area, df["area"])
        df["area"].replace({"Nan": mode_area}, inplace=True)
        df["area"] = df["area"].astype("category")

        num_cols_specs = [
            ("vehicle_count", 0, 10000, "int"),
            ("avg_speed_kmh", 3.0, 120.0, "float"),
            ("accident_count", 0, 20, "int"),
            ("visibility_m", 50, 10000, "int"),
        ]

        for col, min_val, max_val, dtype_name in num_cols_specs:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = np.where(df[col] < min_val, min_val, df[col])
            df[col] = np.where(df[col] > max_val, max_val, df[col])
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

            if dtype_name == "int":
                df[col] = df[col].astype("Int64")
            elif dtype_name == "float":
                df[col] = df[col].astype(float)

        standard_levels = ["Low", "Medium", "High"]
        df["congestion_level"].fillna(
            df["congestion_level"].mode()[0], inplace=True
        )  # Impute NULLs with mode

        df["congestion_level"] = np.where(
            ~df["congestion_level"].isin(standard_levels),
            df["congestion_level"].mode()[0],
            df["congestion_level"],
        )
        df["congestion_level"] = df["congestion_level"].astype("category")

        standard_conditions = ["Dry", "Wet", "Snowy", "Damaged"]
        df["road_condition"].fillna(
            df["road_condition"].mode()[0], inplace=True
        )  # Impute NULLs with mode

        df["road_condition"] = np.where(
            ~df["road_condition"].isin(standard_conditions),
            df["road_condition"].mode()[0],
            df["road_condition"],
        )
        df["road_condition"] = df["road_condition"].astype("category")

        # FIX: Removed redundant pd.to_datetime conversion at the end
        # df['date_time'] = pd.to_datetime(df['date_time'])

        self.log_message("Traffic data cleaning complete.")
        return df.reset_index(drop=True)

    def clean_weather_data(self, df: pd.DataFrame):
        # FIX: Added 'self' argument
        df["weather_id"].fillna(-1, inplace=True)
        df.drop_duplicates(
            subset=df.columns.difference(["weather_id"]), keep="first", inplace=True
        )
        df = df[df["weather_id"] != -1].copy()
        df["weather_id"] = pd.to_numeric(df["weather_id"], errors="coerce").astype(
            "Int64"
        )

        # FIX: Calling the global helper function
        df = clean_date_time(df, "date_time")

        df["city"].fillna("London", inplace=True)
        df["city"] = df["city"].astype(str).str.strip()
        df = df[df["city"].isin(["London"])].copy()
        df["date_time_dt"] = pd.to_datetime(df["date_time"])

        def get_season(dt):
            month = dt.month
            if 3 <= month <= 5:
                return "Spring"
            elif 6 <= month <= 8:
                return "Summer"
            elif 9 <= month <= 11:
                return "Autumn"
            else:
                return "Winter"

        df["season"] = df["date_time_dt"].apply(get_season).astype("category")
        df.drop(columns=["date_time_dt"], inplace=True)

        num_cols_specs = [
            ("temperature_c", 5.0, 35.0, "float"),
            ("humidity", 20, 100, "int"),
            ("rain_mm", 0.0, 100.0, "float"),
            ("wind_speed_kmh", 0.0, 80.0, "float"),
            ("visibility_m", 50, 50000, "int"),
            ("air_pressure_hpa", 950.0, 1050.0, "float"),
        ]

        for col, min_val, max_val, dtype_name in num_cols_specs:
            if col == "air_pressure_hpa" and col not in df.columns:
                df[col] = np.nan

            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = np.where(df[col] < min_val, min_val, df[col])
            df[col] = np.where(df[col] > max_val, max_val, df[col])

            # Use 1000.0 as fill value for air_pressure, median for others
            if col == "air_pressure_hpa":
                fill_val = 1000.0
            else:
                fill_val = df[col].median()

            df[col].fillna(fill_val, inplace=True)

            if dtype_name == "int":
                df[col] = df[col].astype("Int64")
            elif dtype_name == "float":
                df[col] = df[col].astype(float)

        # FIX: Removed redundant cleaning code block for air_pressure_hpa

        standard_conditions = ["Clear", "Rain", "Fog", "Storm", "Snow"]
        df["weather_condition"] = (
            df["weather_condition"].astype(str).str.strip().str.title()
        )
        df["weather_condition"] = np.where(
            ~df["weather_condition"].isin(standard_conditions),
            "Other",
            df["weather_condition"],
        )
        df["weather_condition"].replace(
            {"Other": "Clear", "Nan": "Clear"}, inplace=True
        )
        df["weather_condition"] = df["weather_condition"].astype("category")

        self.log_message("Weather data cleaning complete.")
        return df.reset_index(drop=True)

    def merge_date_city(self, df_weather, df_traffic):
        
        df_weather = df_weather.copy()
        df_traffic = df_traffic.copy()

        
        df_weather["date_time"] = pd.to_datetime(df_weather["date_time"])
        df_traffic["date_time"] = pd.to_datetime(df_traffic["date_time"])

        
        traffic_subset = df_traffic[
            ["date_time", "city", "traffic_id"]
        ]  
        merged_df = pd.merge(
            df_weather, traffic_subset, on=["date_time", "city"], how="left"
        )
        return merged_df

    def _create_silver_layer(self, weather_data, traffic_data):
        """Clean raw CSV files and create silver layer"""
        self.log_message("Cleaning data...")
        try:
            cleaned_traffic = self.clean_traffic_data(traffic_data)
            cleaned_weather = self.clean_weather_data(weather_data)

            merged_data = self.merge_date_city(cleaned_weather, cleaned_traffic)

            
            self.upload_data(merged_data, "silver", "merged_data.parquet")
            self.upload_data(cleaned_weather, "silver", "clean_weather.parquet")
            self.upload_data(cleaned_traffic, "silver", "clean_traffic.parquet")
            self.log_message("✓ Silver layer creation complete.")
            return True

        except Exception as e:
            self.log_message(f"✗ Error in _create_silver_layer: {e}")
            import traceback

            traceback.print_exc()
            raise

    def _create_gold_layer(self):
        try:
            
            self.log_message("Creating weather summary...")
            weather_summary = """
            COPY (
                SELECT 
                    city,
                    COALESCE(season, 'Unknown') as season,
                    COUNT(*) as records_count,
                    ROUND(AVG(temperature_c), 2) as avg_temperature,
                    ROUND(AVG(COALESCE(humidity, 0)), 2) as avg_humidity,
                    ROUND(SUM(COALESCE(rain_mm, 0)), 2) as total_rainfall,
                    MODE(COALESCE(weather_condition, 'Unknown')) as most_common_condition
                FROM 's3://silver/clean_weather.parquet'
                GROUP BY city, COALESCE(season, 'Unknown')
                HAVING COUNT(*) > 0
            ) TO 's3://gold/weather_summary.parquet' (FORMAT 'PARQUET')
            """
            self.con.execute(weather_summary)
            self.log_message("✓ Weather summary created in gold layer")

            
            self.log_message("Creating traffic summary...")
            traffic_summary = """
            COPY (
                SELECT 
                    COALESCE(area, 'Unknown') as area,
                    COALESCE(road_condition, 'Unknown') as road_condition,
                    COUNT(*) as records_count,
                    SUM(COALESCE(vehicle_count, 0)) as total_vehicles,
                    ROUND(AVG(COALESCE(avg_speed_kmh, 0)), 2) as avg_speed,
                    SUM(COALESCE(accident_count, 0)) as total_accidents,
                    ROUND(AVG(COALESCE(visibility_m, 0)), 2) as avg_visibility
                FROM 's3://silver/clean_traffic.parquet'
                GROUP BY COALESCE(area, 'Unknown'), COALESCE(road_condition, 'Unknown')
                HAVING COUNT(*) > 0
            ) TO 's3://gold/traffic_summary.parquet' (FORMAT 'PARQUET')
            """
            self.con.execute(traffic_summary)
            self.log_message("✓ Traffic summary created in gold layer")

            
            self.log_message("Creating combined analysis...")
            combined_analysis = """
            COPY (
                SELECT 
                    DATE_TRUNC('hour', t.date_time) as analysis_hour,
                    COALESCE(t.area, 'Unknown') as area,
                    COALESCE(w.weather_condition, 'Unknown') as weather_condition,
                    COUNT(DISTINCT t.traffic_id) as traffic_measurements,
                    COUNT(DISTINCT w.weather_id) as weather_measurements,
                    ROUND(AVG(w.temperature_c), 2) as avg_temperature,
                    ROUND(AVG(t.avg_speed_kmh), 2) as avg_traffic_speed,
                    SUM(t.accident_count) as total_accidents,
                    ROUND(AVG(t.vehicle_count), 2) as avg_vehicle_count
                FROM 's3://silver/clean_traffic.parquet' t
                LEFT JOIN 's3://silver/clean_weather.parquet' w
                    ON DATE_TRUNC('hour', t.date_time) = DATE_TRUNC('hour', w.date_time)
                    AND t.city = w.city
                WHERE t.date_time IS NOT NULL
                GROUP BY DATE_TRUNC('hour', t.date_time), COALESCE(t.area, 'Unknown'), COALESCE(w.weather_condition, 'Unknown')
                HAVING COUNT(DISTINCT t.traffic_id) > 0
            ) TO 's3://gold/combined_analysis.parquet' (FORMAT 'PARQUET')
            """
            
            self.con.execute(combined_analysis)
            self.log_message("✓ Combined analysis created in gold layer")

            return True

        except Exception as e:
            self.log_message(f"✗ Error in _create_gold_layer: {e}")
            raise

    def generate_weather_data(self, n_rows: int = 5000, city: str = "London"):
        data = {}
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

        # ... (rest of the generate_weather_data logic remains the same)
        temperatures = np.zeros(n_rows)
        for s in ["Winter", "Spring", "Summer", "Autumn"]:
            indices = np.where(seasons_clean == s)[0]
            if s == "Winter":
                temps = np.random.uniform(-5, 15, len(indices))
            elif s == "Spring":
                temps = np.random.uniform(5, 20, len(indices))
            elif s == "Summer":
                temps = np.random.uniform(15, 35, len(indices))
            else:
                temps = np.random.uniform(0, 25, len(indices))
            temperatures[indices] = temps
        outlier_count = int(n_rows * 0.005)
        temperatures[np.random.choice(n_rows, outlier_count, replace=False)] = (
            np.random.choice([-30.0, 60.0], outlier_count)
        )
        temperatures[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = (
            np.nan
        )
        data["temperature_c"] = temperatures.round(2)

        humidity = np.random.randint(20, 101, n_rows)
        outlier_count = int(n_rows * 0.005)
        humidity_outliers = np.random.choice([-10, 150], outlier_count)
        humidity_indices = np.random.choice(n_rows, outlier_count, replace=False)
        humidity[humidity_indices] = humidity_outliers
        humidity_nulls = np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
        humidity = humidity.astype(float)
        humidity[humidity_nulls] = np.nan
        data["humidity"] = humidity

        rain = np.random.rand(n_rows) * 50
        extreme_count = int(n_rows * 0.002)
        rain[np.random.choice(n_rows, extreme_count, replace=False)] = (
            np.random.uniform(120, 150, extreme_count)
        )
        rain[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
        data["rain_mm"] = rain.round(2)

        wind_speed = np.random.rand(n_rows) * 80
        outlier_count = int(n_rows * 0.002)
        wind_speed[np.random.choice(n_rows, outlier_count, replace=False)] = (
            np.random.uniform(200, 250, outlier_count)
        )
        wind_speed[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
        data["wind_speed_kmh"] = wind_speed.round(2)

        visibility_base = np.random.randint(50, 10001, n_rows)
        visibility_messy = visibility_base.astype(object)
        visibility_messy[
            np.random.choice(n_rows, int(n_rows * 0.001), replace=False)
        ] = 50000
        visibility_messy[
            np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
        ] = np.nan
        non_numeric_indices = np.random.choice(
            n_rows, int(n_rows * 0.01), replace=False
        )
        visibility_messy[non_numeric_indices] = np.random.choice(
            ["low", "unknown", "N/A", "high"], len(non_numeric_indices)
        )
        data["visibility_m"] = visibility_messy

        conditions = ["Clear", "Rain", "Fog", "Storm", "Snow"]
        weather_condition = np.random.choice(
            conditions, n_rows, p=[0.4, 0.2, 0.15, 0.05, 0.2]
        )
        weather_condition[
            np.random.choice(n_rows, int(n_rows * 0.03), replace=False)
        ] = np.nan
        data["weather_condition"] = weather_condition

        air_pressure = np.random.uniform(950, 1050, n_rows)
        air_pressure[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = (
            np.nan
        )
        data["air_pressure_hpa"] = air_pressure.round(2)

        city_data = np.full(n_rows, city, dtype=object)
        city_data[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
        city_data[np.random.choice(n_rows, int(n_rows * 0.01), replace=False)] = (
            "unknown"
        )
        data["city"] = city_data

        df = pd.DataFrame(data)
        df["weather_condition"] = df["weather_condition"].astype("category")

        date_times_messy = base_dates.astype(str)
        indices_v1 = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
        date_times_messy[indices_v1] = [
            d.strftime("%d/%m/%Y %H:%M") for d in base_dates[indices_v1]
        ]
        indices_v2 = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
        date_times_messy[indices_v2] = [
            d.strftime("%Y-%m-%dT%H:%M") for d in base_dates[indices_v2]
        ]
        date_times_messy[
            np.random.choice(n_rows, int(n_rows * 0.01), replace=False)
        ] = "25/:61"
        date_times_messy[
            np.random.choice(n_rows, int(n_rows * 0.01), replace=False)
        ] = "2099-13-40"
        date_times_messy[
            np.random.choice(n_rows, int(n_rows * 0.01), replace=False)
        ] = np.nan
        date_times_messy[
            np.random.choice(n_rows, int(n_rows * 0.01), replace=False)
        ] = "unknown"
        df.insert(1, "date_time", date_times_messy)

        seasons_messy = seasons_clean.astype(object)
        seasons_messy[np.random.choice(n_rows, int(n_rows * 0.03), replace=False)] = (
            np.nan
        )
        df.insert(4, "season", seasons_messy)
        df["season"] = df["season"].astype("category")

        base_ids = np.arange(5001, 5001 + n_rows)
        weather_ids = base_ids.astype(float)
        duplicate_indices = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
        weather_ids[duplicate_indices] = np.random.choice(
            base_ids[:1000], len(duplicate_indices)
        )
        weather_ids[np.random.choice(n_rows, int(n_rows * 0.005), replace=False)] = (
            np.nan
        )
        df.insert(0, "weather_id", pd.Series(weather_ids).astype(pd.Int64Dtype()))

        df["humidity"] = df["humidity"].astype(pd.Int64Dtype())

        logging.info("Data generation complete.")
        print(df.head())
        return df

    def generate_traffic_data(self, n_rows: int = 5000, city: str = "London"):
        # FIX: Added 'self' argument. Logic remains the same.
        data = {}
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2024, 12, 31)
        time_range_seconds = int((end_date - start_date).total_seconds())

        random_seconds = np.random.randint(0, time_range_seconds, n_rows)
        base_dates = np.array(
            [start_date + timedelta(seconds=int(s)) for s in random_seconds]
        )

        areas = [
            "Camden",
            "Chelsea",
            "Islington",
            "Southwark",
            "Kensington",
            "Westminster",
        ]
        area_weights = [0.15, 0.20, 0.15, 0.15, 0.25, 0.10]
        clean_areas = np.random.choice(areas, n_rows, p=area_weights)
        area_data = clean_areas.astype(object)
        area_data[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
        data["area"] = area_data

        # ... (rest of the generate_traffic_data logic remains the same)
        vehicle_counts = np.zeros(n_rows)
        for area in areas:
            indices = np.where(clean_areas == area)[0]
            if area in ["Chelsea", "Westminster", "Kensington"]:
                counts = np.random.randint(100, 5001, len(indices))
            else:
                counts = np.random.randint(0, 3001, len(indices))
            vehicle_counts[indices] = counts

        outlier_count = int(n_rows * 0.003)
        vehicle_counts[np.random.choice(n_rows, outlier_count, replace=False)] = (
            np.random.randint(20000, 30000, outlier_count)
        )

        vehicle_counts_nulls = np.random.choice(
            n_rows, int(n_rows * 0.02), replace=False
        )
        vehicle_counts = vehicle_counts.astype(float)
        vehicle_counts[vehicle_counts_nulls] = np.nan
        data["vehicle_count"] = vehicle_counts

        avg_speeds = np.random.uniform(3, 120, n_rows)
        high_traffic_indices = np.where(vehicle_counts > 3000)[0]
        avg_speeds[high_traffic_indices] = np.random.uniform(
            3, 30, len(high_traffic_indices)
        )
        low_traffic_indices = np.where(vehicle_counts < 500)[0]
        avg_speeds[low_traffic_indices] = np.random.uniform(
            40, 100, len(low_traffic_indices)
        )
        invalid_count = int(n_rows * 0.005)
        avg_speeds[np.random.choice(n_rows, invalid_count, replace=False)] = (
            np.random.uniform(-10.0, -1.0, invalid_count)
        )
        avg_speeds[np.random.choice(n_rows, int(n_rows * 0.02), replace=False)] = np.nan
        data["avg_speed_kmh"] = avg_speeds.round(2)

        accident_counts = np.random.poisson(lam=1, size=n_rows)
        accident_counts[accident_counts > 10] = 10
        high_risk_indices = np.where((vehicle_counts > 4000) | (avg_speeds < 10))[0]
        accident_counts[high_risk_indices] = np.random.poisson(
            lam=2, size=len(high_risk_indices)
        )
        extreme_count = int(n_rows * 0.001)
        accident_counts[np.random.choice(n_rows, extreme_count, replace=False)] = (
            np.random.randint(50, 70, extreme_count)
        )
        accident_counts_nulls = np.random.choice(
            n_rows, int(n_rows * 0.02), replace=False
        )
        accident_counts = accident_counts.astype(float)
        accident_counts[accident_counts_nulls] = np.nan
        data["accident_count"] = accident_counts

        congestion = np.full(n_rows, "Medium", dtype=object)
        high_indices = np.where((vehicle_counts > 4500) | (avg_speeds < 15))[0]
        congestion[high_indices] = "High"
        low_indices = np.where((vehicle_counts < 1000) & (avg_speeds > 60))[0]
        congestion[low_indices] = "Low"
        congestion[np.random.choice(n_rows, int(n_rows * 0.03), replace=False)] = np.nan
        data["congestion_level"] = congestion

        conditions = ["Dry", "Wet", "Snowy", "Damaged"]
        road_condition = np.random.choice(conditions, n_rows, p=[0.7, 0.2, 0.05, 0.05])
        road_condition[np.random.choice(n_rows, int(n_rows * 0.03), replace=False)] = (
            np.nan
        )
        data["road_condition"] = road_condition

        visibility = np.random.randint(50, 10001, n_rows)
        visibility_nulls = np.random.choice(n_rows, int(n_rows * 0.02), replace=False)
        visibility = visibility.astype(float)
        visibility[visibility_nulls] = np.nan
        data["visibility_m"] = visibility

        city_data = np.full(n_rows, city, dtype=object)
        city_data[np.random.choice(n_rows, int(n_rows * 0.01), replace=False)] = np.nan
        data["city"] = city_data
        df = pd.DataFrame(data)

        df["congestion_level"] = df["congestion_level"].astype("category")
        df["road_condition"] = df["road_condition"].astype("category")

        date_times_messy = base_dates.astype(str)
        indices_v1 = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
        date_times_messy[indices_v1] = [
            d.strftime("%d/%m/%Y %H:%M") for d in base_dates[indices_v1]
        ]
        indices_v2 = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
        date_times_messy[indices_v2] = [
            d.strftime("%Y-%m-%dT%H:%M:00") for d in base_dates[indices_v2]
        ]
        date_times_messy[
            np.random.choice(n_rows, int(n_rows * 0.005), replace=False)
        ] = "TBD"
        date_times_messy[
            np.random.choice(n_rows, int(n_rows * 0.005), replace=False)
        ] = "2099-00-00 99:99"
        date_times_messy[
            np.random.choice(n_rows, int(n_rows * 0.01), replace=False)
        ] = np.nan
        df.insert(1, "date_time", date_times_messy)

        base_ids = np.arange(9001, 9001 + n_rows)
        traffic_ids = base_ids.astype(float)
        duplicate_indices = np.random.choice(n_rows, int(n_rows * 0.1), replace=False)
        traffic_ids[duplicate_indices] = np.random.choice(
            base_ids[:1000], len(duplicate_indices)
        )
        traffic_ids[np.random.choice(n_rows, int(n_rows * 0.005), replace=False)] = (
            np.nan
        )
        df.insert(0, "traffic_id", traffic_ids)

        df["traffic_id"] = df["traffic_id"].astype("Int64")
        df["vehicle_count"] = df["vehicle_count"].astype("Int64")
        df["accident_count"] = df["accident_count"].astype("Int64")
        df["visibility_m"] = df["visibility_m"].astype("Int64")

        logging.info("Data generation complete.")
        print(df.head())
        return df


def main():
    """Main execution function"""
    MINIO_URL = "http://localhost:9000"
    ACCESS_KEY = "root"
    SECRET_KEY = "root442002"

    try:
        # Initialize DataLakeManager
        lake = DataLakeManager(MINIO_URL, ACCESS_KEY, SECRET_KEY)

        # Generate synthetic data
        weather_df = lake.generate_weather_data(500, "London")
        traffic_df = lake.generate_traffic_data(500, "London")

        print(weather_df.head())
        print(traffic_df.head())

        # Upload to bronze layer (as parquet for efficiency/consistency)
        lake.upload_data(weather_df, "bronze", "raw_weather.parquet")
        lake.upload_data(traffic_df, "bronze", "raw_traffic.parquet")

        # FIX: Call run_etl_pipeline() which contains the full logic
        success = lake.run_etl_pipeline()

        if success:
            print("\n" + "=" * 50)
            print("ETL Pipeline Completed Successfully!")
            print("=" * 50)

            # List files in each bucket to verify
            for bucket in ["bronze", "silver", "gold"]:
                try:
                    response = lake.s3.list_objects_v2(Bucket=bucket)
                    files = [obj["Key"] for obj in response.get("Contents", [])]
                    print(f"\nFiles in {bucket} bucket:")
                    for file in files:
                        print(f"  - {file}")
                except Exception as e:
                    print(f"\nError listing {bucket} bucket: {e}")
        else:
            print("\n" + "=" * 50)
            print("ETL Pipeline Failed!")
            print("=" * 50)

    except Exception as e:
        print(f"\nFatal error in main execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
