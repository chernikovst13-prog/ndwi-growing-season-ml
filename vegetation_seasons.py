import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
import warnings
import re

try:
    from numpy.exceptions import RankWarning
except ImportError:
    try:
        from numpy import RankWarning
    except ImportError:
        RankWarning = UserWarning

warnings.simplefilter('ignore', RankWarning)


def folder_grafics_TEMP(input_folder, output_folder, n, summary_filename="all_temp.txt"):
    """Processes reference txt files with temperatures and returns the temp_db dictionary."""
    temp_db = {}
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} not found. Skipping temperatures reading.")
        return temp_db
        
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    summary_file_path = os.path.join(output_folder, summary_filename)
    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    
    if not txt_files:
        print(f"No txt temperature files in folder {input_folder}")
        return temp_db
        
    with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
        for file_path in txt_files:
            filename = os.path.basename(file_path)
            raw_data = []
            station = "Unknown"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip().split()
                    if not s or s[0] == "Index":
                        continue
                    try:
                        station = s[0]
                        dates_val = f"{s[1]}-{s[2].zfill(2)}-{s[3].zfill(2)}"
                        temp_val = float(s[5])
                        raw_data.append((dates_val, temp_val))
                    except (IndexError, ValueError):
                        continue
                        
            if not raw_data:
                continue
                
            raw_data.sort(key=lambda x: x[0])
            dates_str = [x[0] for x in raw_data]
            dates_dt = pd.to_datetime(dates_str)
            temps = [x[1] for x in raw_data]
            total = len(temps)
            full_answer = [station]
            
            averaged_temp = []
            for i in range(total):
                startidx = max(0, i - n)
                endidx = min(total, i + n + 1)
                window = temps[startidx:endidx]
                averaged_temp.append(sum(window) / len(window))

            station_key = re.sub(r'\D', '', filename)
            if not station_key: 
                station_key = filename.replace('.txt', '').strip().lower()

            for d_str, t_raw, t_smooth in zip(dates_str, temps, averaged_temp):
                temp_db[(station_key, d_str)] = (t_raw, t_smooth)

            # Determining vegetation by real temperature (for reference)
            for j in range(1, total - 4):
                if averaged_temp[j-1] <= 5 and averaged_temp[j] >= 5 and averaged_temp[j+4] >= 5:
                    full_answer.append(dates_str[j])
                    break
            for j in range(1, total - 4):
                if averaged_temp[j-1] <= 10 and averaged_temp[j] >= 10 and averaged_temp[j+4] >= 5:
                    full_answer.append(dates_str[j])
                    break
            for i in range(total - 2, 4, -1):
                if averaged_temp[i+1] <= 5 and averaged_temp[i] >= 5 and averaged_temp[i-4] >= 5:
                    full_answer.append(dates_str[i])
                    break
            for i in range(total - 2, 4, -1):
                if averaged_temp[i+1] <= 10 and averaged_temp[i] >= 10 and averaged_temp[i-4] >= 5:
                    full_answer.append(dates_str[i])
                    break
                    
            summary_file.write(f"{filename}: {full_answer}\n")
            
    print(f"trainig temperatures processing completed. Found points: {len(temp_db)}")
    return temp_db


def extract_ndwi_features(input_folder, temp_db=None):
    """Reads CSV NDWI files and generates features for ML."""
    if not os.path.exists(input_folder):
        print(f"Folder {input_folder} not found. Skipping.")
        return pd.DataFrame()
        
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files in folder {input_folder}.")
        return pd.DataFrame()
        
    all_stations_features = []
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        station_key = re.sub(r'\D', '', file_name)
        if not station_key:
            station_key = file_name.replace('.csv', '').strip().lower()
        
        df = pd.read_csv(file_path, names=['date', 'NDWI'], header=0)
        df = df.dropna(subset=['date'])
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date']).set_index('date')
        
        if df.empty:
            continue
        
        df = df.groupby(level = 0).mean()

        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(full_range)
        
        df['NDWI'] = pd.to_numeric(df['NDWI'], errors='coerce').interpolate(method='linear').bfill().ffill()
        df = df.reset_index().rename(columns={'index': 'date'})
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        df['Day_of_year'] = df['date'].dt.dayofyear

        if temp_db:
            def get_temps_for_row(row):
                key = (station_key, row['date_str'])
                if key in temp_db:
                    return temp_db[key]
                return np.nan, np.nan
            df[['Temp_Raw', 'Temp_Smoothed']] = df.apply(get_temps_for_row, axis=1, result_type='expand')
            df['Temp_Raw'] = df['Temp_Raw'].interpolate(method='linear').bfill().ffill().fillna(0.0)
            df['Temp_Smoothed'] = df['Temp_Smoothed'].interpolate(method='linear').bfill().ffill().fillna(0.0)
        else:
            # No temperatures for new locations
            df['Temp_Raw'] = 0.0
            df['Temp_Smoothed'] = 0.0

        def get_slope(y):
            if len(y) > 1:
                x = np.arange(len(y))
                return np.polyfit(x, y, deg=1)[0]
            return 0.0
        
        df['last_angle_5'] = df['NDWI'].rolling(window=5, min_periods=2).apply(get_slope, raw=True).shift(1)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
        df['next_angle_5'] = df['NDWI'].shift(-1).rolling(window=indexer, min_periods=2).apply(get_slope, raw=True)
        df['trend_last_direction'] = (df['last_angle_5'] > 0).astype(int)
        df['trend_next_direction'] = (df['next_angle_5'] > 0).astype(int)
        
        if len(df) < 50:
            continue

        temps = df['NDWI'].values
        total = len(temps)
        
        x_idx = np.arange(total)
        poly_coefic_all = np.polyfit(x_idx, temps, deg=4)
        poly_func_all = np.poly1d(poly_coefic_all)
        trends_value_all = poly_func_all(x_idx)
        
        df['Original'] = df['NDWI']
        df['Trend_4'] = trends_value_all
        df['Difference'] = df['NDWI'].diff().fillna(0)
        df['File_name'] = file_name
        
        df['Is_transition'] = 0 
        
        all_stations_features.append(df[[
            'File_name', 'date', 'date_str', 'Original', 'Trend_4', 'Difference', 'Day_of_year',
            'last_angle_5', 'next_angle_5', 'trend_last_direction', 'trend_next_direction',
            'Temp_Raw', 'Temp_Smoothed', 'Is_transition'
        ]])

    if all_stations_features:
        res_df = pd.concat(all_stations_features, ignore_index=True)
        res_df['last_angle_5'] = res_df['last_angle_5'].fillna(0.0)
        res_df['next_angle_5'] = res_df['next_angle_5'].fillna(0.0)
        print (res_df)
        return res_df
    else:
        print ("no alf")
    return pd.DataFrame()


def calculate_vegetation_dates(temps, dates_str):
    """Reliable temperature method for determining dates of crossing 5 and 10 degrees."""
    total = len(temps)
    dates = []
    
    for j in range(1, total - 4):
        if temps[j-1] <= 5 and temps[j] >= 5 and temps[j+4] >= 5:
            dates.append(dates_str[j])
            break
            
    for j in range(1, total - 4):
        if temps[j-1] <= 10 and temps[j] >= 10 and temps[j+4] >= 5:
            dates.append(dates_str[j])
            break
            
    for i in range(total - 2, 4, -1):
        if temps[i+1] <= 5 and temps[i] >= 5 and temps[i-4] >= 5:
            dates.append(dates_str[i])
            break
            
    for i in range(total - 2, 4, -1):
        if temps[i+1] <= 10 and temps[i] >= 10 and temps[i-4] >= 5:
            dates.append(dates_str[i])
            break
            
    return sorted(list(set(dates)))


def run_pipeline(train_df, test_df, output_folder, training_data_path):
    """Trains the model on Petrun data and predicts vegetation ."""
    if train_df.empty:
        print("Error: training dataset is empty!")
        return

    train_df.to_csv(training_data_path, index=False)
    print(f"Historical dataset saved to: {training_data_path}")
    
    feature_cols = [
        'Original', 'Trend_4', 'Difference', 'Day_of_year', 
        'last_angle_5', 'next_angle_5', 'trend_last_direction', 'trend_next_direction'
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df['Temp_Smoothed']
    
    print("Training RandomForestRegressor model on Petrun data...")
    model_reg = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model_reg.fit(X_train, y_train)
    print("Model successfully trained!")

    if test_df.empty:
        print("No new files for testing found. Finishing work.")
        return

    print("Predicting temperatures and calculating vegetation for new data...")
    test_df['Predicted_Temp'] = model_reg.predict(test_df[feature_cols])
    test_df['Pred_Transition'] = 0 # Initialize column for tracking transitions
    
    results_file_path = os.path.join(output_folder, "predicted_vegetation_seasons.txt")
    
    with open(results_file_path, 'w', encoding='utf-8') as f_out:
        for file_name, group in test_df.groupby('File_name'):
            group_sorted = group.sort_values('date')
            pred_temps = group_sorted['Predicted_Temp'].values
            dates_str = group_sorted['date_str'].values
            dates_dt = group_sorted['date']
            
            veg_dates = calculate_vegetation_dates(pred_temps, dates_str)
            
            # Mark the found dates in our dataframe column
            test_df.loc[(test_df['File_name'] == file_name) & (test_df['date_str'].isin(veg_dates)), 'Pred_Transition'] = 1
            
            f_out.write(f"{file_name}: {veg_dates}\n")
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates_dt, pred_temps, color='darkred', linewidth=2, label='Predicted Temp (Smoothed)')
            plt.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='5°C Threshold')
            plt.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='10°C Threshold')
            
            for vd in veg_dates:
                vd_dt = pd.to_datetime(vd)
                plt.axvline(x=vd_dt, color='blue', linestyle=':', alpha=0.8)
                plt.text(vd_dt, min(pred_temps) + 2, vd, rotation=90, color='blue', fontsize=9)
            
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.title(f"Predicted Temp & Vegetation Seasons for {file_name}")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_folder, file_name.replace('.csv', '_predicted_temp.png')))
            plt.close()
            
    print(f"Vegetation season results written to: {results_file_path}")

    detailed_output_path = os.path.join(output_folder, "ml_predicted_details.txt")
    print("Generating detailed text file with all daily values...")
    
    with open(detailed_output_path, 'w', encoding='utf-8') as f_det:
        for idx, row in test_df.iterrows():
            line = (f"Date: {row['date_str']} | "
                    f"NDWI: {row['Original']:.4f} | "
                    f"Trend_4: {row['Trend_4']:.4f} | "
                    f"Diff: {row['Difference']:.4f} | "
                    f"Temp_Raw: {row['Temp_Raw']:.1f} | "
                    f"Temp_Smooth: {row['Temp_Smoothed']:.1f} | "
                    f"Pred_Temp_Smooth: {row['Predicted_Temp']:.1f} | "
                    f"DayOfYear: {row['Day_of_year']} | "
                    f"Is_Transition: {row['Is_transition']} | "
                    f"Pred_Transition: {row['Pred_Transition']}")
            f_det.write(line + "\n")
            
    print(f"Detailed daily predictions file created successfully: {detailed_output_path}")


if __name__ == "__main__":
    # path settings
    train_temp_dir = r""   
    train_ndwi_dir = r""   
    test_ndwi_dir = r""  
    output_directory = r""
    ml_training_file = os.path.join(output_directory, "my_training_data.csv")
    
    smoothing_window = int(input("Enter smoothing window for Petrun temperatures (n): "))
    
    print("\n--- Step 1: Collecting and preparing reference database (training) ---")
    temp_db_train = folder_grafics_TEMP(train_temp_dir, output_directory, smoothing_window, "petrun_all_temp.txt")
    train_df = extract_ndwi_features(train_ndwi_dir, temp_db=temp_db_train)
    
    print("\n--- Step 2: Preparing new data (Naryan-Mar) ---")
    test_df = extract_ndwi_features(test_ndwi_dir, temp_db=None)
    
    print("\n--- Step 3: Machine learning and vegetation prediction ---")
    run_pipeline(train_df, test_df, output_directory, ml_training_file)
    
    print("\n all processes have been completed!")
