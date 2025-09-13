# ===============================================

# combine.py
# 1. 將所有檔案以指定數量[FILES_PER_GROUP]合併成一張時域圖 -> 能夠觀察三軸震動變化的規律性
# 2. 將所有檔案統一繪製成一張時域圖
# 3. 對合併後的數據進行 FFT 分析，找出主要頻率成分，
#    並將能量大小 > peak_threshold的頻率數值在圖上做標記並輸出成文字檔 (peak_threshold可調整)

# (建議的參數設定)[資料來源]             [檔案數][SENSOR_TYPE][FILES_PER_GROUP][Z_CONTACT][INTERVAL]
# sensor_on_6Axis\500ms\20250807163010 -> 25       6Axis         4             500        0.5
# sensor_on_6Axis\1500ms\20250807163311 -> 51      6Axis         6             1500       1.5
# sensor_on_Probe\500ms\20250807165504 -> 25       Probe         4             500        0.5
# sensor_on_Probe\1500ms\20250807165057 -> 51      Probe         6             1500       1.5
# 3.2
# =================================================

import os, glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ========== 路徑 & 檔名規定 ==========
DATA_DIR     = r"D:\1_project\newData\Raw_Data_20250822\高架地板\sensor on probe\1500ms\20250820150650"   # ← 改這裡：資料夾
OPT_DATA_DIR     = r"D:\1_project\newData\Raw_Data_20250822\高架地板\sensor on probe\1500ms"   # ← 改這裡：資料夾
FILE_PATTERN = "test_*_Raw_*.csv"         # ← 改這裡：檔名樣式（也可用 *.csv）
OUTPUT_DIR  = r"D:\1_project\TEST_3\Probe_1500"  # ← 改這裡：清洗後 CSV 輸出
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ========== 參數設定 ==========
# 採樣頻率 (Hz)
SAMPLING_RATE = 51200
# 設定要合併幾個檔案的數據成一張圖
FILES_PER_GROUP = 8
SENSOR_TYPE = "Probe"  # "6Axis" or "Probe"
Z_CONTACT = "1500"      # Stop or 500 or 1500
INTERVAL = 1.5  # 虛線間隔 (秒)

def extract_raw_num(filename):
    match = re.search(r'Raw_(\d+)', filename)
    return int(match.group(1)) if match else -1

files = sorted(
    glob.glob(os.path.join(DATA_DIR, "test_*_Raw_*.csv")),
    key=lambda x: extract_raw_num(x)
)
if not files:
    print(f"在 {DATA_DIR} 中沒有找到符合 {FILE_PATTERN} 的檔案。")
else:
    print(f"找到 {len(files)} 個檔案。")

num_groups = len(files) // FILES_PER_GROUP
remainder_files = len(files) % FILES_PER_GROUP
print(f"num_groups: {num_groups}, remainder_files: {remainder_files}")

opt_file = pd.read_csv(os.path.join(OPT_DATA_DIR, "_20250820150650.csv"))
opt_df_T = pd.DataFrame()
opt_df_T["Time(S)"] = opt_file["Time(ms)"] / 1000.0  # 換算成秒

def plot_all_amplitude(vib_df, opt_df, out_path):

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    fig.suptitle(f'Overall Time Domain for {SENSOR_TYPE}_{Z_CONTACT}', fontsize=16)

    axes[0].plot(vib_df["X_Time(S)"], vib_df["X_Sensor_Amplitude(V)"], color="red")
    axes[0].set_title("X-axis Amplitude")

    axes[1].plot(vib_df["X_Time(S)"], vib_df["Y_Sensor_Amplitude(V)"], color="green")
    axes[1].set_title("Y-axis Amplitude")

    axes[2].plot(vib_df["X_Time(S)"], vib_df["Z_Sensor_Amplitude(V)"], color="blue")
    axes[2].set_title("Z-axis Amplitude")

    axes[3].plot(opt_df_T["Time(S)"], opt_df["Optical_Power(dBm)"], color="orange")
    axes[3].set_title("Optical Power (dBm)")
    axes[3].set_xlabel("Time (s)")


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"整體時域圖已儲存至: {out_path}")

# --- 區間比較 (每 4 秒) ---
def plot_segmented(vib_df, opt_df, segment_length, out_dir="segments"):
    os.makedirs(out_dir, exist_ok=True)
    max_time = min(vib_df["X_Time(S)"].max(), opt_df_T["Time(S)"].max())
    print(f"最大時間範圍: {max_time:.2f} 秒")

    seg_id = 1
    start_t = 0
    while start_t < max_time:
        end_t = start_t + segment_length

        vib_seg = vib_df[(vib_df["X_Time(S)"] >= start_t) & (vib_df["X_Time(S)"] < end_t)]
        opt_seg = opt_df[((opt_df["Time(ms)"]/1000.0) >= start_t) & ((opt_df["Time(ms)"]/1000.0) < end_t)]

        if vib_seg.empty or opt_seg.empty:
            start_t = end_t
            seg_id += 1
            continue

        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        fig.suptitle(f"Segment {seg_id}: {start_t:.1f}–{end_t:.1f} s", fontsize=16)

        axes[0].plot(vib_seg["X_Time(S)"], vib_seg["X_Sensor_Amplitude(V)"], color="red")
        axes[0].set_ylabel("X")

        axes[1].plot(vib_seg["X_Time(S)"], vib_seg["Y_Sensor_Amplitude(V)"], color="green")
        axes[1].set_ylabel("Y")

        axes[2].plot(vib_seg["X_Time(S)"], vib_seg["Z_Sensor_Amplitude(V)"], color="blue")
        axes[2].set_ylabel("Z")

        axes[3].plot(opt_seg["Time(ms)"]/1000.0, opt_seg["Optical_Power(dBm)"], color="orange")
        axes[3].set_ylabel("Optical")
        axes[3].set_xlabel("Time (s)")        
     
        # --- 只在前三個子圖畫虛線 ---
        interval = INTERVAL  # 與你的主程式設定一致
        if interval and interval > 0:
            min_time_seg = vib_seg['X_Time(S)'].iloc[0]
            max_time_seg = vib_seg['X_Time(S)'].iloc[-1]
            line_times = np.arange(np.ceil(min_time_seg/interval) * interval, max_time_seg, interval)
            for t in line_times:
                for ax in [axes[0], axes[1], axes[2]]:
                    ax.axvline(x=t, color='grey', linestyle='--', linewidth=3.0, alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        seg_out = os.path.join(out_dir, f"segment_{seg_id}.png")
        plt.savefig(seg_out, dpi=300)
        plt.close(fig)
        print(f"區間圖 {seg_id} 已儲存至: {seg_out}")
        # 移動到下一個區間"""
        start_t = end_t
        seg_id += 1

def plot_and_save_group(group_files, group_index, group_type, start, end):
    """
    讀取一個群組的 CSV 檔案，將它們串接並繪製成一張圖，然後儲存。
    
    Args:
        group_files (list): 包含要處理的 CSV 檔案路徑的列表。
        group_index (int): 群組編號。
        group_type (str): 群組類型，可以是 'Group' 或 'Remainder'。
    """
    group_dfs = []
    
    for file_path in group_files:
        try:
            df = pd.read_csv(file_path)
            
            # 確保有 'time' 欄位，如果沒有，修改這裡
            if 'X_Time(S)' not in df.columns:
                print(f"警告: 檔案 {file_path} 中沒有 'time' 欄位，已跳過。")
                continue
                
            # 將每個檔案的時間軸疊加上去
            if group_dfs:
                last_time = group_dfs[-1]['X_Time(S)'].iloc[-1]
                df['X_Time(S)'] = df['X_Time(S)'] + last_time

            # 假設振幅欄位名稱為 'x', 'y', 'z'，如果不同修改這裡
            if 'X_Sensor_Amplitude(V)' not in df.columns or \
                'Y_Sensor_Amplitude(V)' not in df.columns or \
                'Z_Sensor_Amplitude(V)' not in df.columns:
                 print(f"警告: 檔案 {file_path} 中缺少振幅欄位，已跳過。")
                 continue
            
            group_dfs.append(pd.DataFrame({
                'X_Time(S)': df['X_Time(S)'],
                'x': df['X_Sensor_Amplitude(V)'],
                'y': df['Y_Sensor_Amplitude(V)'],
                'z': df['Z_Sensor_Amplitude(V)'],
            }))
            
        except FileNotFoundError:
            print(f"警告: 找不到檔案 {file_path}，已跳過。")
        except Exception as e:
            print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
            
    if not group_dfs:
        print(f"群組 {group_index} 中沒有可用的數據，已跳過繪圖。")
        return
        
    combined_df = pd.concat(group_dfs, ignore_index=True)
    
    # 建立一個新的圖，包含三個子圖 (X, Y, Z)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # 繪製 X 軸數據
    ax1.plot(combined_df['X_Time(S)'], combined_df['x'], color='red')
    ax1.set_title('X-axis Amplitude')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    
    # 繪製 Y 軸數據
    ax2.plot(combined_df['X_Time(S)'], combined_df['y'], color='green')
    ax2.set_title('Y-axis Amplitude')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    
    # 繪製 Z 軸數據
    ax3.plot(combined_df['X_Time(S)'], combined_df['z'], color='blue')
    ax3.set_title('Z-axis Amplitude')
    ax3.set_ylabel('Amplitude')
    ax3.set_xlabel('Time')
    ax3.grid(True)
    
    # --- 在圖上標示虛線 ---
    # 獲取時間軸的範圍
    min_time = combined_df['X_Time(S)'].iloc[0]
    max_time = combined_df['X_Time(S)'].iloc[-1]
    
    # 設定虛線的間隔
    interval = INTERVAL
    
    # 生成虛線的時間點
    # np.arange(start, stop, step) 函數
    line_times = np.arange(np.ceil(min_time/interval) * interval, max_time, interval)
    
    # 繪製虛線
    for t in line_times:
        # 在每個子圖上都繪製虛線
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=t, color='grey', linestyle='--', linewidth=3.0, alpha=0.7)
            
    # 設定整個圖的總標題
    fig.suptitle(f'Time Domain Plot of  {SENSOR_TYPE}_{Z_CONTACT}: FILE {start} - {end}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 調整標題位置
    
    # 儲存圖表，檔名包含群組編號
    output_filename = f'FILE {start} - {end}.png'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300)
    print(f"圖表已儲存至：{output_path}")
"""
# 處理每個群組的檔案，並單獨繪圖和儲存
for i in range(num_groups):
    start_index = i * FILES_PER_GROUP
    end_index = start_index + FILES_PER_GROUP
    group_files = files[start_index:end_index]
    print(f"處理群組 {i + 1}，包含檔案：{group_files}")
    # 調用繪圖函式
    plot_and_save_group(group_files, i + 1, 'Group', start_index, end_index)

# 處理最後剩餘的檔案 (如果有)
if remainder_files > 0:
    start_index = num_groups * FILES_PER_GROUP
    group_files = files[start_index:]
    # 調用繪圖函式，標記為「剩餘檔案」
    plot_and_save_group(group_files, num_groups + 1, 'Remainder', start_index, len(files))
print("\n--- 所有群組處理完成 ---")
"""
# 讀取並合併所有檔案
all_dfs = []
last_time_all = 0
for file_path in files:
    try:
        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in ['X_Time(S)', 'X_Sensor_Amplitude(V)', 'Y_Sensor_Amplitude(V)', 'Z_Sensor_Amplitude(V)']):
            print(f"警告: 檔案 {file_path} 缺少必要的欄位，已跳過。")
            continue
            
        df['X_Time(S)'] = df['X_Time(S)'] + last_time_all
        last_time_all = df['X_Time(S)'].iloc[-1]
        
        all_dfs.append(df)
        
    except Exception as e:
        print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
        continue

if all_dfs:
    combined_df_total = pd.concat(all_dfs, ignore_index=True)

    # 繪製整體圖
    overall_out = os.path.join(OUTPUT_DIR, f"overall_XYZ_Optical_{SENSOR_TYPE}_{Z_CONTACT}.png")
    plot_all_amplitude(combined_df_total, opt_file, overall_out)

    # 繪製每 4 秒一張的比較圖
    plot_segmented(combined_df_total, opt_file, FILES_PER_GROUP,
                   out_dir=os.path.join(OUTPUT_DIR, "segments"))
    """
    print("繪製總體時域圖")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'Overall Time Domain Plot of {SENSOR_TYPE}_{Z_CONTACT}', fontsize=16)
    ax1.plot(combined_df_total['X_Time(S)'], combined_df_total['X_Sensor_Amplitude(V)'], color='red')
    ax1.set_title('X-axis Amplitude')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax2.plot(combined_df_total['X_Time(S)'], combined_df_total['Y_Sensor_Amplitude(V)'], color='green')
    ax2.set_title('Y-axis Amplitude')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax3.plot(combined_df_total['X_Time(S)'], combined_df_total['Z_Sensor_Amplitude(V)'], color='blue')
    ax3.set_title('Z-axis Amplitude')
    ax3.set_ylabel('Amplitude')
    ax3.set_xlabel('Time')
    ax3.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    overall_output_filename = f'overall_time_domain_plot_{SENSOR_TYPE}_{Z_CONTACT}.png'
    overall_output_path = os.path.join(OUTPUT_DIR, overall_output_filename)
    plt.savefig(overall_output_path, dpi=300)
    print(f"總體時域圖已儲存至：{overall_output_path}")
    plt.close(fig)
    """
    print("\n--- 開始繪製總體頻域圖 ---")
    # 執行 FFT 運算
    n = len(combined_df_total)
    T = 1.0 / SAMPLING_RATE
    
    # 計算頻率軸
    xf = np.fft.fftfreq(n, T)[:n//2]
    
    # 對每個軸進行 FFT
    yf_x = np.fft.fft(combined_df_total['X_Sensor_Amplitude(V)'])
    yf_y = np.fft.fft(combined_df_total['Y_Sensor_Amplitude(V)'])
    yf_z = np.fft.fft(combined_df_total['Z_Sensor_Amplitude(V)'])
    
    # 計算振幅的絕對值，並正規化
    amplitude_x = 2.0/n * np.abs(yf_x[0:n//2])
    amplitude_y = 2.0/n * np.abs(yf_y[0:n//2])
    amplitude_z = 2.0/n * np.abs(yf_z[0:n//2])
    
    # 設定一個閾值，只考慮振幅超過此值的尖峰
    # 你可以根據你的圖來調整這個值
    peak_threshold = 0.0005 

    # 尋找 X 軸的尖峰
    # distance=1000 確保相鄰尖峰之間有足夠的距離，避免找到太多雜訊
    peaks_x, _ = find_peaks(amplitude_x, height=peak_threshold, distance=1000)
    print("\nX 軸尖峰頻率與振幅:")
    for peak_index in peaks_x:
        print(f"頻率: {xf[peak_index]:.2f} Hz, 振幅: {amplitude_x[peak_index]:.4f}")

    # 尋找 Y 軸的尖峰
    peaks_y, _ = find_peaks(amplitude_y, height=peak_threshold, distance=1000)
    print("\nY 軸尖峰頻率與振幅:")
    for peak_index in peaks_y:
        print(f"頻率: {xf[peak_index]:.2f} Hz, 振幅: {amplitude_y[peak_index]:.4f}")

    # 尋找 Z 軸的尖峰
    peaks_z, _ = find_peaks(amplitude_z, height=peak_threshold, distance=1000)
    print("\nZ 軸尖峰頻率與振幅:")
    for peak_index in peaks_z:
        print(f"頻率: {xf[peak_index]:.2f} Hz, 振幅: {amplitude_z[peak_index]:.4f}")
    
    # 將三軸尖峰資訊寫入 TXT 檔案
    txt_filename = f"{SENSOR_TYPE}_{Z_CONTACT}ms_frequency_peaks.txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("threshold: {:.4f}\n\n".format(peak_threshold))
        f.write("X 軸尖峰頻率與振幅:\n")
        for peak_index in peaks_x:
            f.write(f"頻率: {xf[peak_index]:.2f} Hz, 振幅: {amplitude_x[peak_index]:.4f}\n")
        f.write("\nY 軸尖峰頻率與振幅:\n")
        for peak_index in peaks_y:
            f.write(f"頻率: {xf[peak_index]:.2f} Hz, 振幅: {amplitude_y[peak_index]:.4f}\n")
        f.write("\nZ 軸尖峰頻率與振幅:\n")
        for peak_index in peaks_z:
            f.write(f"頻率: {xf[peak_index]:.2f} Hz, 振幅: {amplitude_z[peak_index]:.4f}\n")
    print(f"尖峰頻率與振幅已儲存至：{txt_path}")

    # 繪製總體頻域圖
    fig_fft, (ax1_fft, ax2_fft, ax3_fft) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig_fft.suptitle(f'Overall Frequency Domain Plot (FFT) OF {SENSOR_TYPE}_{Z_CONTACT}', fontsize=16)

    ax1_fft.plot(xf, amplitude_x, color='red')
    ax1_fft.plot(xf[peaks_x], amplitude_x[peaks_x], "x", color='black', label="Peaks") # 在尖峰位置繪製 'x' 標記
    ax1_fft.set_title('X-axis Frequency Spectrum')
    ax1_fft.set_ylabel('Amplitude')
    ax1_fft.grid(True)
    ax1_fft.legend() # 顯示圖例

    ax2_fft.plot(xf, amplitude_y, color='green')
    ax2_fft.plot(xf[peaks_y], amplitude_y[peaks_y], "x", color='black', label="Peaks")
    ax2_fft.set_title('Y-axis Frequency Spectrum')
    ax2_fft.set_ylabel('Amplitude')
    ax2_fft.grid(True)
    ax2_fft.legend()

    ax3_fft.plot(xf, amplitude_z, color='blue')
    ax3_fft.plot(xf[peaks_z], amplitude_z[peaks_z], "x", color='black', label="Peaks")
    ax3_fft.set_title('Z-axis Frequency Spectrum')
    ax3_fft.set_ylabel('Amplitude')
    ax3_fft.set_xlabel('Frequency (Hz)')
    ax3_fft.grid(True)
    ax3_fft.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 儲存頻域圖
    fft_output_filename = f'overall_frequency_domain_plot_{SENSOR_TYPE}_{Z_CONTACT}.png'
    fft_output_path = os.path.join(OUTPUT_DIR, fft_output_filename)
    plt.savefig(fft_output_path, dpi=300)
    print(f"總體頻域圖已儲存至：{fft_output_path}")

    plt.close(fig_fft)

else:
    print("沒有足夠的有效數據來繪製總體頻域圖。")

print("所有圖表繪製完成。")