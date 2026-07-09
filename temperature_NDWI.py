import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import csv
from datetime import datetime

def folder_grafics_NDVI(input_folder, output_folder, n):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    summary_file_path = os.path.join(output_folder, "all_ndvi.txt")
    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
   
    if not txt_files:
        print("there are no txt files in NDVI folder")
        return
   
    with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
        for file_path in txt_files:
            filename = os.path.basename(file_path)
           
            raw_data = []
            station = "Unknown"
           
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    s = line.strip().split()
                    if not s or s[0] == "Индекс":
                        continue
                    try:
                        station = s[0]
                        dates_val = f"{s[1]}-{s[2].zfill(2)}-{s[3].zfill(2)}"
                        temp_val = float(s[5])
                        raw_data.append((dates_val, temp_val))
                    except (IndexError, ValueError):
                        continue
           
            if not raw_data:
                print(f"file {filename} skipped (no data)")
                continue
               
            raw_data.sort(key=lambda x: x[0])
            dates = [x[0] for x in raw_data]
            temps = [x[1] for x in raw_data]
           
            full_answer = [station]
            total = len(temps)
           
            # --- Сглаживание ---
            averaged_temp = []
            for i in range(total):
                startidx = max(0, i - n)
                endidx = min(total, i + n + 1)
                window = temps[startidx:endidx]
                averaged_temp.append(sum(window) / len(window))

            # --- Поиск фенофаз ---
            for j in range(1, total - 4):
                if averaged_temp[j-1] <= 5 and averaged_temp[j] >= 5 and averaged_temp[j+4] >= 5:
                    full_answer.append(dates[j])
                    break
            for j in range(1, total - 4):
                if averaged_temp[j-1] <= 10 and averaged_temp[j] >= 10 and averaged_temp[j+4] >= 5:
                    full_answer.append(dates[j])
                    break
            for i in range(total - 2, 4, -1):
                if averaged_temp[i+1] <= 5 and averaged_temp[i] >= 5 and averaged_temp[i-4] >= 5:
                    full_answer.append(dates[i])
                    break
            for i in range(total - 2, 4, -1):
                if averaged_temp[i+1] <= 10 and averaged_temp[i] >= 10 and averaged_temp[i-4] >= 5:
                    full_answer.append(dates[i])
                    break
           
            summary_file.write(f"{filename}: {full_answer}\n")
           
            # --- График ---
            plt.figure(figsize=(12, 6))
            plt.plot(dates, temps, alpha=0.3, label='Исходная темп.', color='red')
            plt.plot(dates, averaged_temp, alpha=0.8, label=f'Сглаженная (n={n})', color='blue', linewidth=2)
            plt.axhline(y=5, color='green', linestyle='--', linewidth=1.5, label='Порог 5°C')
            plt.axhline(y=10, color='darkgreen', linestyle='--', linewidth=1.5, label='Порог 10°C')

            x_indx = np.arange(total)
            poly_coeficients = np.polyfit(x_indx, temps, deg=3)
            polyfunction = np.poly1d(poly_coeficients)
            plt.plot(dates, polyfunction(x_indx), color='black', linestyle='-.', linewidth=2, label='trend')
           
            plt.xticks(dates[::30], rotation=45)
            plt.title(f'NDVI: Анализ температурных трендов ({filename})')
            plt.xlabel('Дата')
            plt.ylabel('Температура (°C)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()

            graph_filename = filename.replace('.txt', '.png')
            plt.savefig(os.path.join(output_folder, graph_filename))
            plt.close()
    print("NDVI processing done.")


def folder_grafics_NDWI(input_folder, output_folder, n):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
       
    summary_file_path = os.path.join(output_folder, "all_ndwi.txt")
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
   
    if not csv_files:
        print("there are no csv files :(")
        return
       
    with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            raw_data = []
           
            station = file_name.replace('.csv', '')
           
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=',')
                header = next(reader, None)  
               
                for row in reader:
                    if not row or len(row) < 2:
                        continue
                   
                    raw_date = row[0].strip()
                    raw_val = row[1].strip()
                   
                    if not raw_val:
                        continue
                       
                    try:
                        date_obj = datetime.strptime(raw_date, "%b %d, %Y")
                        dates_val = date_obj.strftime("%Y-%m-%d")
                       
                        temp_val = float(raw_val)
                        raw_data.append((dates_val, temp_val)) 
                    except (ValueError, IndexError):
                        continue
                       
            if not raw_data:
                print(f"file {file_name} skipped (no valid data)")
                continue
               
            raw_data.sort(key=lambda x: x[0])
            dates = [x[0] for x in raw_data]
            temps = [x[1] for x in raw_data]
           
            full_answer = [station]
            total = len(temps)
               
            x_idx = np.arange(total)
            poly_coefic_all = np.polyfit(x_idx, temps, deg=4)
            poly_func_all = np.poly1d(poly_coefic_all)
            trends_value_all = poly_func_all(x_idx)
           
            min1_idx_raw = np.argmin(temps)
            
            exclusion_zone = 45 
            temps_masked = np.array(temps, dtype=float)
            start_mask = max(0, min1_idx_raw - exclusion_zone)
            end_mask = min(total, min1_idx_raw + exclusion_zone + 1)
            temps_masked[start_mask:end_mask] = np.inf
            

            min2_idx_raw = np.argmin(temps_masked)

            start_slice = min(min1_idx_raw, min2_idx_raw)
            end_slice = max(min1_idx_raw, min2_idx_raw)

            full_answer.append(dates[start_slice])
            full_answer.append(dates[end_slice])

            x_slice = np.arange(start_slice, end_slice + 1)
            temps_slice = np.array(temps[start_slice:end_slice + 1])
            dates_slice = dates[start_slice:end_slice + 1]
           
            coef_lin = np.polyfit(x_slice, temps_slice, deg=1)
            y_lin = np.poly1d(coef_lin)(x_slice)

            coef_poly3 = np.polyfit(x_slice, temps_slice, deg=3)
            y_poly = np.poly1d(coef_poly3)(x_slice)

            difference = y_lin - y_poly
            intersect_indices = []

            for k in range(len(difference) - 1):
                if difference[k] * difference[k+1] <= 0:
                    actual_idx = x_slice[k]
                    if actual_idx not in intersect_indices:
                        intersect_indices.append(actual_idx)
                        full_answer.append(dates[actual_idx])
                   
            summary_file.write(f"{file_name}: {full_answer}\n")

            plt.figure(figsize=(13, 7))
            plt.plot(dates, temps, alpha=0.4, label='Исходные данные', color='gray')
            plt.plot(dates, trends_value_all, alpha=0.4, label='Общий тренд (deg=4)', color='cyan', linestyle=':')
           
            if len(dates_slice) > 0:
                plt.plot(dates_slice, y_lin, color='orange', linewidth=2.5, linestyle='--', label='Линейный тренд')
                plt.plot(dates_slice, y_poly, color='purple', linewidth=2.5, label='Кубический тренд')
           
            plt.scatter([dates[start_slice], dates[end_slice]], [temps[start_slice], temps[end_slice]],
                        color='black', s=120, zorder=5, label='Выявленные минимумы')
           
            max_temp = max(temps) if temps else 1
            min_temp = min(temps) if temps else 0
            for idx_int in intersect_indices:
                d_int = dates[idx_int]
                y_int = temps[idx_int]
               
                plt.scatter(d_int, y_int, color='red', marker='X', s=150, zorder=6)
                plt.annotate(
                    f'Пересечение\n{d_int}',
                    xy=(d_int, y_int),
                    xytext=(d_int, y_int + ((max_temp - min_temp) * 0.1)),
                    arrowprops=dict(facecolor='red', shrink=0.08, width=1, headwidth=6, headlength=6),
                    ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
                )
           
            plt.xticks(dates[::30], rotation=45)
            plt.title(f'NDWI: Анализ минимумов и трендов ({file_name})') 
            plt.xlabel('Дата')
            plt.ylabel('Значение')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='upper right')
            plt.tight_layout()

            graph_filename = file_name.replace('.csv', '.png')
            plt.savefig(os.path.join(output_folder, graph_filename))
            plt.close()
           
    print("NDWI processing done.")


if __name__ == "__main__":
    #Path to the folder with temperature files
    input_directory_v = r""
    #Path to the folder with NDWI index files
    input_directory_w = r""
    #Path to the folder where you want to save obtained data 
    output_directory = r"" 
   
    smoothing_window = int(input("Введите окно усреднения (n): "))
   
    folder_grafics_NDVI(input_directory_v, output_directory, smoothing_window)
    folder_grafics_NDWI(input_directory_w, output_directory, smoothing_window)
    print("all done")