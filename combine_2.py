# ===============================================

# combine_2.py
# 1. 針對濾波過後的數據，將多個檔案依序合併，繪製三軸的時域圖，並將原始數據與濾波後的數據疊加顯示
# 2. 對合併後的數據進行 FFT 分析，找出主要頻率成分，
#    並將能量大小 > peak_threshold 的頻率數值在圖上做標記並輸出成文字檔 (peak_threshold可調整 )
# 3. 

# 檔案格式: "test_*_Raw_*_clean_X.csv" 、"test_*_Raw_*_clean_Y.csv"、"test_*_Raw_*_clean_Z.csv"
# 檔案內容: t_s, raw, clean 三欄位
# 輸出圖檔: overall_time_domain_plot_XXX.png, overall_frequency_domain_plot_XXX.png
# 輸出文字檔: XXX_frequency_peaks.txt
# 參數設定: DATA_DIR, OUTPUT_DIR, SAMPLING_RATE, SENSOR_TYPE, Z_CONTACT, peak_threshold

# =================================================

import os
# ========== 路徑 & 檔名規定 ==========
DATA_DIR     = r"D:\1_project\TEST\Probe_1500"   # ← 改這裡：資料夾
OUTPUT_DIR  = r"D:\1_project\TEST\test"  # ← 改這裡：清洗後 CSV 輸出
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ========== 參數設定 ==========
# 採樣頻率 (Hz)
SAMPLING_RATE = 51200
SENSOR_TYPE = "Probe"  # "6Axis" or "Probe"
Z_CONTACT = "1500"      # Stop or 500 or 1500
INTERVAL = 1.5  # 時域圖上虛線間隔 (秒) 根據數據修改
FILES_PER_GROUP = 6  # 每幾個檔案合併成一張圖 (可調整)

import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
# ===================== 工具函式 =====================
def extract_raw_num(filename):
    match = re.search(r'Raw_(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_files():
    # 取得所有 clean_X, clean_Y, clean_Z 檔案
    clean_x_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "test_*_Raw_*_clean_X.csv")),
        key=lambda x: extract_raw_num(x)
    )
    clean_y_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "test_*_Raw_*_clean_Y.csv")),
        key=lambda x: extract_raw_num(x)
    )
    clean_z_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "test_*_Raw_*_clean_Z.csv")),
        key=lambda x: extract_raw_num(x)
    )

    # 以 Raw_* 編號配對 X/Y/Z 檔案
    def get_raw_num_from_path(path):
        m = re.search(r'Raw_(\d+)', path)
        return int(m.group(1)) if m else -1

    raw_nums = set(get_raw_num_from_path(f) for f in clean_x_files) & \
               set(get_raw_num_from_path(f) for f in clean_y_files) & \
               set(get_raw_num_from_path(f) for f in clean_z_files)
    raw_nums = sorted(raw_nums)

    if not raw_nums:
        print(f"在 {DATA_DIR} 中沒有找到完整的 X/Y/Z clean 檔案組。")
        return
    return raw_nums, clean_x_files, clean_y_files, clean_z_files

def find_frequency_peaks(amplitude, xf, threshold=0.0005, distance=1000):
    peaks, _ = find_peaks(amplitude, height=threshold, distance=distance)
    return [(xf[i], amplitude[i]) for i in peaks]

def export_peaks_txt(peaks_x, peaks_y, peaks_z, sensor_type, z_contact, output_dir, threshold):
    txt_filename = f"{sensor_type}_{z_contact}ms_frequency_peaks.txt"
    txt_path = os.path.join(output_dir, txt_filename)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("threshold: {:.4f}\n\n".format(threshold))
        f.write("X 軸尖峰頻率與振幅:\n")
        for freq, amp in peaks_x:
            f.write(f"頻率: {freq:.2f} Hz, 振幅: {amp:.4f}\n")
        f.write("\nY 軸尖峰頻率與振幅:\n")
        for freq, amp in peaks_y:
            f.write(f"頻率: {freq:.2f} Hz, 振幅: {amp:.4f}\n")
        f.write("\nZ 軸尖峰頻率與振幅:\n")
        for freq, amp in peaks_z:
            f.write(f"頻率: {freq:.2f} Hz, 振幅: {amp:.4f}\n")
    print(f"尖峰頻率與振幅已儲存至：{txt_path}")


def plot_all_together():

    raw_nums, clean_x_files, clean_y_files, clean_z_files = load_files()

    # 合併所有 Raw_* 的 X/Y/Z clean 檔案
    all_clean_x = []
    all_clean_y = []
    all_clean_z = []
    last_time_x = 0
    last_time_y = 0
    last_time_z = 0

    for raw_num in raw_nums:
        x_file = next(f for f in clean_x_files if f'Raw_{raw_num}' in f)
        y_file = next(f for f in clean_y_files if f'Raw_{raw_num}' in f)
        z_file = next(f for f in clean_z_files if f'Raw_{raw_num}' in f)

        try:
            df_x = pd.read_csv(x_file)
            df_y = pd.read_csv(y_file)
            df_z = pd.read_csv(z_file)
            # 更改：檢查必要欄位是否存在
            if not all(col in df_x.columns for col in ['t_s', 'raw', 'clean']):
                print(f"警告: 檔案 {x_file} 缺少必要的欄位，已跳過。")
                continue
            if not all(col in df_y.columns for col in ['t_s', 'raw', 'clean']):
                print(f"警告: 檔案 {y_file} 缺少必要的欄位，已跳過。")
                continue
            if not all(col in df_z.columns for col in ['t_s', 'raw', 'clean']):
                print(f"警告: 檔案 {z_file} 缺少必要的欄位，已跳過。")
                continue

            df_x['t_s'] = df_x['t_s'] + last_time_x
            last_time_x = df_x['t_s'].iloc[-1]
            all_clean_x.append(df_x)

            df_y['t_s'] = df_y['t_s'] + last_time_y
            last_time_y = df_y['t_s'].iloc[-1]
            all_clean_y.append(df_y)

            df_z['t_s'] = df_z['t_s'] + last_time_z
            last_time_z = df_z['t_s'].iloc[-1]
            all_clean_z.append(df_z)
        except Exception as e:
            print(f"讀取 Raw_{raw_num} 檔案時發生錯誤: {e}")

    if not all_clean_x or not all_clean_y or not all_clean_z:
        print("沒有足夠的 clean 數據。")
        return

    combined_x = pd.concat(all_clean_x, ignore_index=True)
    combined_y = pd.concat(all_clean_y, ignore_index=True)
    combined_z = pd.concat(all_clean_z, ignore_index=True)

    # 統整時域圖
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axs[0].plot(combined_x['t_s'], combined_x['raw'], color='darkred', label='X raw', alpha=0.6)
    axs[0].plot(combined_x['t_s'], combined_x['clean'], color='red', label='X clean')
    axs[0].set_title(f'{SENSOR_TYPE}_X-axis Clean Amplitude')
    axs[0].set_ylabel('Amplitude')
    axs[0].grid(True)
    axs[0].legend(loc="upper right")
    axs[1].plot(combined_y['t_s'], combined_y['raw'], color='darkgreen', label='Y raw', alpha=0.6)
    axs[1].plot(combined_y['t_s'], combined_y['clean'], color='green', label='Y clean')
    axs[1].set_title(f'{SENSOR_TYPE}_Y-axis Clean Amplitude')
    axs[1].set_ylabel('Amplitude')
    axs[1].grid(True)
    axs[1].legend(loc="upper right")
    axs[2].plot(combined_z['t_s'], combined_z['raw'], color='navy', label='Z raw', alpha=0.4)
    axs[2].plot(combined_z['t_s'], combined_z['clean'], color='blue', label='Z clean')
    axs[2].set_title(f'{SENSOR_TYPE}_Z-axis Clean Amplitude')
    axs[2].set_ylabel('Amplitude')
    axs[2].set_xlabel('Time(s)')
    axs[2].grid(True)
    axs[2].legend(loc="upper right")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_filename = f'overall_time_domain_plot_{SENSOR_TYPE}_{Z_CONTACT}_clean_XYZ.png'
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    # 統整頻域圖
    peak_threshold = 0.0005

    def get_fft_and_peaks(df, axis_color):
        n = len(df)
        T = 1.0 / SAMPLING_RATE
        xf = np.fft.fftfreq(n, T)[:n//2]
        yf = np.fft.fft(df['clean'])
        amplitude = 2.0/n * np.abs(yf[0:n//2])
        peaks = find_frequency_peaks(amplitude, xf, peak_threshold)
        return xf, amplitude, peaks

    xf_x, amp_x, peaks_x = get_fft_and_peaks(combined_x, 'blue')
    xf_y, amp_y, peaks_y = get_fft_and_peaks(combined_y, 'green')
    xf_z, amp_z, peaks_z = get_fft_and_peaks(combined_z, 'red')

    fig_fft, axs_fft = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axs_fft[0].plot(xf_x, amp_x, color='red')
    axs_fft[0].plot([f for f, _ in peaks_x], [a for _, a in peaks_x], "x", color='black', label="Peaks")
    axs_fft[0].set_title(f'{SENSOR_TYPE}_X-axis Frequency Spectrum (Clean)')
    axs_fft[0].set_ylabel('Amplitude')
    axs_fft[0].grid(True)
    axs_fft[0].legend()
    axs_fft[1].plot(xf_y, amp_y, color='green')
    axs_fft[1].plot([f for f, _ in peaks_y], [a for _, a in peaks_y], "x", color='black', label="Peaks")
    axs_fft[1].set_title(f'{SENSOR_TYPE}_Y-axis Frequency Spectrum (Clean)')
    axs_fft[1].set_ylabel('Amplitude')
    axs_fft[1].grid(True)
    axs_fft[1].legend()
    axs_fft[2].plot(xf_z, amp_z, color='blue')
    axs_fft[2].plot([f for f, _ in peaks_z], [a for _, a in peaks_z], "x", color='black', label="Peaks")
    axs_fft[2].set_title(f'{SENSOR_TYPE}_Z-axis Frequency Spectrum (Clean)')
    axs_fft[2].set_ylabel('Amplitude')
    axs_fft[2].set_xlabel('Frequency (Hz)')
    axs_fft[2].grid(True)
    axs_fft[2].legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fft_output_filename = f'overall_frequency_domain_plot_{SENSOR_TYPE}_{Z_CONTACT}_clean_XYZ.png'
    fft_output_path = os.path.join(OUTPUT_DIR, fft_output_filename)
    plt.savefig(fft_output_path, dpi=300)
    plt.close(fig_fft)

    # 尖峰輸出
    export_peaks_txt(peaks_x, peaks_y, peaks_z, SENSOR_TYPE, f"{Z_CONTACT}_clean_XYZ", OUTPUT_DIR, peak_threshold)


"""
def plot_in_group(raw_nums, x_files, y_files, z_files, group_index, start, end):

    # 合併所有 Raw_* 的 X/Y/Z clean 檔案
    all_clean_x = []
    all_clean_y = []
    all_clean_z = []
    last_time_x = 0
    last_time_y = 0
    last_time_z = 0

    for raw_num in raw_nums:
        x_file = next(f for f in clean_x_files if f'Raw_{raw_num}' in f)
        y_file = next(f for f in clean_y_files if f'Raw_{raw_num}' in f)
        z_file = next(f for f in clean_z_files if f'Raw_{raw_num}' in f)

        try:
            df_x = pd.read_csv(x_file)
            df_y = pd.read_csv(y_file)
            df_z = pd.read_csv(z_file)
            # 更改：檢查必要欄位是否存在
            if not all(col in df_x.columns for col in ['t_s', 'raw', 'clean']):
                print(f"警告: 檔案 {x_file} 缺少必要的欄位，已跳過。")
                continue
            if not all(col in df_y.columns for col in ['t_s', 'raw', 'clean']):
                print(f"警告: 檔案 {y_file} 缺少必要的欄位，已跳過。")
                continue
            if not all(col in df_z.columns for col in ['t_s', 'raw', 'clean']):
                print(f"警告: 檔案 {z_file} 缺少必要的欄位，已跳過。")
                continue

            df_x['t_s'] = df_x['t_s'] + last_time_x
            last_time_x = df_x['t_s'].iloc[-1]
            all_clean_x.append(df_x)

            df_y['t_s'] = df_y['t_s'] + last_time_y
            last_time_y = df_y['t_s'].iloc[-1]
            all_clean_y.append(df_y)

            df_z['t_s'] = df_z['t_s'] + last_time_z
            last_time_z = df_z['t_s'].iloc[-1]
            all_clean_z.append(df_z)
        except Exception as e:
            print(f"讀取 Raw_{raw_num} 檔案時發生錯誤: {e}")

    if not all_clean_x or not all_clean_y or not all_clean_z:
        print("沒有足夠的 clean 數據。")
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
if __name__ == "__main__":
    plot_all_together()
    #print("\n--- 開始處理各群組檔案並繪圖 ---")
    """
    raw_nums, clean_x_files, clean_y_files, clean_z_files = load_files()
    num_groups = len(clean_x_files) // FILES_PER_GROUP
    remainder_files = len(clean_x_files) % FILES_PER_GROUP
    print(f"num_groups: {num_groups}, remainder_files: {remainder_files}")

    # 處理每個群組的檔案，並單獨繪圖和儲存
    for i in range(num_groups):
        start_index = i * FILES_PER_GROUP
        end_index = start_index + FILES_PER_GROUP
        group_x_files = clean_x_files[start_index:end_index]
        group_y_files = clean_y_files[start_index:end_index]
        group_z_files = clean_z_files[start_index:end_index]
        print(f"處理群組 {i + 1}")
        # 調用繪圖函式
        plot_in_group(raw_nums, group_x_files, group_y_files, group_z_files, i + 1, 'Group', start_index, end_index)

    # 處理最後剩餘的檔案 (如果有)
    if remainder_files > 0:
        start_index = num_groups * FILES_PER_GROUP
        group_x_files = clean_x_files[start_index:end_index]
        group_y_files = clean_y_files[start_index:end_index]
        group_z_files = clean_z_files[start_index:end_index]
        # 調用繪圖函式，標記為「剩餘檔案」
        plot_in_group(raw_nums, group_x_files, group_y_files, group_z_files, num_groups + 1, 'Remainder', start_index, len(files))
    print("\n--- 所有群組處理完成 ---")
    """
    print("所有分析與圖表繪製完成。")