import os
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ===================== 使用者互動設定 =====================
def get_user_settings():
    print("請選擇分析模式：")
    print("1. 移動檔案分析 (分檔案時域圖、總體時域圖、總體頻域圖、頻率段統計)")
    print("2. 靜止檔案分析 (clean與原始資料總體時域圖、總體頻域圖、頻率段統計)")
    mode = input("請輸入模式編號 (1 或 2)：").strip()
    if mode == "1":
        data_dir = input("請輸入原始資料資料夾路徑：").strip()
        output_dir = input("請輸入輸出資料夾路徑：").strip()
        file_pattern = input("請輸入檔案樣式 (如 test_*_Raw_*.csv)：").strip()
        sensor_type = input("請輸入 SENSOR_TYPE (如 Probe 或 6Axis)：").strip()
        z_contact = input("請輸入 Z_CONTACT (如 500 或 1500)：").strip()
        sampling_rate = int(input("請輸入採樣頻率 (如 51200)：").strip())
        files_per_group = int(input("請輸入每組檔案數 (如 8)：").strip())
        interval = float(input("請輸入虛線間隔秒數 (如 0.5)：").strip())
        return {
            "mode": "move",
            "data_dir": data_dir,
            "output_dir": output_dir,
            "file_pattern": file_pattern,
            "sensor_type": sensor_type,
            "z_contact": z_contact,
            "sampling_rate": sampling_rate,
            "files_per_group": files_per_group,
            "interval": interval
        }
    elif mode == "2":
        clean_dir = r"D:\1_project\TEST\Probe_500_____2"  # clean資料夾路徑寫死
        output_dir = input("請輸入輸出資料夾路徑：").strip()
        sensor_type = input("請輸入 SENSOR_TYPE (如 Probe)：").strip()
        z_contact = input("請輸入 Z_CONTACT (如 Stop)：").strip()
        sampling_rate = int(input("請輸入採樣頻率 (如 51200)：").strip())
        return {
            "mode": "static",
            "clean_dir": clean_dir,
            "output_dir": output_dir,
            "sensor_type": sensor_type,
            "z_contact": z_contact,
            "sampling_rate": sampling_rate
        }
    else:
        print("輸入錯誤，請重新執行。")
        exit(1)

# ===================== 工具函式 =====================
def extract_raw_num(filename):
    match = re.search(r'Raw_(\d+)', filename)
    return int(match.group(1)) if match else -1

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

# ===================== 移動檔案分析 =====================
def move_file_analysis(settings):
    DATA_DIR = settings["data_dir"]
    OUTPUT_DIR = settings["output_dir"]
    FILE_PATTERN = settings["file_pattern"]
    SENSOR_TYPE = settings["sensor_type"]
    Z_CONTACT = settings["z_contact"]
    SAMPLING_RATE = settings["sampling_rate"]
    FILES_PER_GROUP = settings["files_per_group"]
    INTERVAL = settings["interval"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(DATA_DIR, FILE_PATTERN)),
        key=lambda x: extract_raw_num(x)
    )
    if not files:
        print(f"在 {DATA_DIR} 中沒有找到符合 {FILE_PATTERN} 的檔案。")
        return

    # 分檔案畫出時域圖
    def plot_and_save_group(group_files, group_index, group_type, start, end):
        group_dfs = []
        for file_path in group_files:
            try:
                df = pd.read_csv(file_path)
                if 'X_Time(S)' not in df.columns:
                    print(f"警告: 檔案 {file_path} 中沒有 'X_Time(S)' 欄位，已跳過。")
                    continue
                if group_dfs:
                    last_time = group_dfs[-1]['X_Time(S)'].iloc[-1]
                    df['X_Time(S)'] = df['X_Time(S)'] + last_time
                if not all(col in df.columns for col in ['X_Sensor_Amplitude(V)', 'Y_Sensor_Amplitude(V)', 'Z_Sensor_Amplitude(V)']):
                    print(f"警告: 檔案 {file_path} 中缺少振幅欄位，已跳過。")
                    continue
                group_dfs.append(pd.DataFrame({
                    'X_Time(S)': df['X_Time(S)'],
                    'x': df['X_Sensor_Amplitude(V)'],
                    'y': df['Y_Sensor_Amplitude(V)'],
                    'z': df['Z_Sensor_Amplitude(V)'],
                }))
            except Exception as e:
                print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
        if not group_dfs:
            print(f"群組 {group_index} 中沒有可用的數據，已跳過繪圖。")
            return
        combined_df = pd.concat(group_dfs, ignore_index=True)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        ax1.plot(combined_df['X_Time(S)'], combined_df['x'], color='red')
        ax1.set_title('X-axis Amplitude')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True)
        ax2.plot(combined_df['X_Time(S)'], combined_df['y'], color='green')
        ax2.set_title('Y-axis Amplitude')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        ax3.plot(combined_df['X_Time(S)'], combined_df['z'], color='blue')
        ax3.set_title('Z-axis Amplitude')
        ax3.set_ylabel('Amplitude')
        ax3.set_xlabel('Time')
        ax3.grid(True)
        min_time = combined_df['X_Time(S)'].iloc[0]
        max_time = combined_df['X_Time(S)'].iloc[-1]
        line_times = np.arange(np.ceil(min_time/INTERVAL) * INTERVAL, max_time, INTERVAL)
        for t in line_times:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=t, color='grey', linestyle='--', linewidth=3.0, alpha=0.7)
        fig.suptitle(f'Time Domain Plot of {SENSOR_TYPE}_{Z_CONTACT}: FILE {start} - {end}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = f'FILE {start} - {end}.png'
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(output_path, dpi=300)
        plt.close(fig)

    num_groups = len(files) // FILES_PER_GROUP
    remainder_files = len(files) % FILES_PER_GROUP
    for i in range(num_groups):
        start_index = i * FILES_PER_GROUP
        end_index = start_index + FILES_PER_GROUP
        group_files = files[start_index:end_index]
        plot_and_save_group(group_files, i + 1, 'Group', start_index, end_index)
    if remainder_files > 0:
        start_index = num_groups * FILES_PER_GROUP
        group_files = files[start_index:]
        plot_and_save_group(group_files, num_groups + 1, 'Remainder', start_index, len(files))

    # 合併所有檔案
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
    if not all_dfs:
        print("沒有足夠的有效數據來繪製總體圖。")
        return
    combined_df_total = pd.concat(all_dfs, ignore_index=True)

    # 畫出總體時域圖
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
    plt.close(fig)

    # 劃出總體頻域圖並找尖峰
    n = len(combined_df_total)
    T = 1.0 / SAMPLING_RATE
    xf = np.fft.fftfreq(n, T)[:n//2]
    yf_x = np.fft.fft(combined_df_total['X_Sensor_Amplitude(V)'])
    yf_y = np.fft.fft(combined_df_total['Y_Sensor_Amplitude(V)'])
    yf_z = np.fft.fft(combined_df_total['Z_Sensor_Amplitude(V)'])
    amplitude_x = 2.0/n * np.abs(yf_x[0:n//2])
    amplitude_y = 2.0/n * np.abs(yf_y[0:n//2])
    amplitude_z = 2.0/n * np.abs(yf_z[0:n//2])
    peak_threshold = 0.0005
    peaks_x = find_frequency_peaks(amplitude_x, xf, peak_threshold)
    peaks_y = find_frequency_peaks(amplitude_y, xf, peak_threshold)
    peaks_z = find_frequency_peaks(amplitude_z, xf, peak_threshold)
    export_peaks_txt(peaks_x, peaks_y, peaks_z, SENSOR_TYPE, Z_CONTACT, OUTPUT_DIR, peak_threshold)

    fig_fft, (ax1_fft, ax2_fft, ax3_fft) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig_fft.suptitle(f'Overall Frequency Domain Plot (FFT) OF {SENSOR_TYPE}_{Z_CONTACT}', fontsize=16)
    ax1_fft.plot(xf, amplitude_x, color='red')
    ax1_fft.plot([f for f, _ in peaks_x], [a for _, a in peaks_x], "x", color='black', label="Peaks")
    ax1_fft.set_title('X-axis Frequency Spectrum')
    ax1_fft.set_ylabel('Amplitude')
    ax1_fft.grid(True)
    ax1_fft.legend()
    ax2_fft.plot(xf, amplitude_y, color='green')
    ax2_fft.plot([f for f, _ in peaks_y], [a for _, a in peaks_y], "x", color='black', label="Peaks")
    ax2_fft.set_title('Y-axis Frequency Spectrum')
    ax2_fft.set_ylabel('Amplitude')
    ax2_fft.grid(True)
    ax2_fft.legend()
    ax3_fft.plot(xf, amplitude_z, color='blue')
    ax3_fft.plot([f for f, _ in peaks_z], [a for _, a in peaks_z], "x", color='black', label="Peaks")
    ax3_fft.set_title('Z-axis Frequency Spectrum')
    ax3_fft.set_ylabel('Amplitude')
    ax3_fft.set_xlabel('Frequency (Hz)')
    ax3_fft.grid(True)
    ax3_fft.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fft_output_filename = f'overall_frequency_domain_plot_{SENSOR_TYPE}_{Z_CONTACT}.png'
    fft_output_path = os.path.join(OUTPUT_DIR, fft_output_filename)
    plt.savefig(fft_output_path, dpi=300)
    plt.close(fig_fft)

# ===================== 靜止檔案分析 =====================
def static_file_analysis(settings):
    CLEAN_DIR = settings["clean_dir"]
    OUTPUT_DIR = settings["output_dir"]
    SENSOR_TYPE = settings["sensor_type"]
    Z_CONTACT = settings["z_contact"]
    SAMPLING_RATE = settings["sampling_rate"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # clean資料
    clean_files = sorted(
        glob.glob(os.path.join(CLEAN_DIR, "test_*_Raw_*_clean_Z.csv")),
        key=lambda x: extract_raw_num(x)
    )
    if not clean_files:
        print(f"在 {CLEAN_DIR} 中沒有找到 clean 檔案。")
        return

    # 合併 clean 檔案
    all_clean_dfs = []
    last_time_clean = 0
    for file_path in clean_files:
        try:
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in ['t_s', 'clean']):
                print(f"警告: 檔案 {file_path} 缺少必要的欄位，已跳過。")
                continue
            df['t_s'] = df['t_s'] + last_time_clean
            last_time_clean = df['t_s'].iloc[-1]
            all_clean_dfs.append(df)
        except Exception as e:
            print(f"讀取檔案 {file_path} 時發生錯誤: {e}")
    if not all_clean_dfs:
        print("沒有足夠的 clean 數據。")
        return
    combined_clean_df = pd.concat(all_clean_dfs, ignore_index=True)

    # 畫出 clean 與原始資料總體時域圖
    fig, ax_clean = plt.subplots(figsize=(15, 5), sharex=True)
    ax_clean.plot(combined_clean_df['t_s'], combined_clean_df['clean'], color='blue', label='Clean Z')
    ax_clean.set_title(f'{SENSOR_TYPE}_Z-axis Clean Amplitude')
    ax_clean.set_ylabel('Amplitude')
    ax_clean.set_xlabel('Time(s)')
    ax_clean.grid(True)
    ax_clean.legend()
    overall_output_filename = f'overall_time_domain_plot_{SENSOR_TYPE}_{Z_CONTACT}_clean_Z.png'
    overall_output_path = os.path.join(OUTPUT_DIR, overall_output_filename)
    plt.savefig(overall_output_path, dpi=300)
    plt.close(fig)

    # 劃出 clean 總體頻域圖並找尖峰
    n = len(combined_clean_df)
    T = 1.0 / SAMPLING_RATE
    xf = np.fft.fftfreq(n, T)[:n//2]
    yf_z = np.fft.fft(combined_clean_df['clean'])
    amplitude_z = 2.0/n * np.abs(yf_z[0:n//2])
    peak_threshold = 0.0005
    peaks_z = find_frequency_peaks(amplitude_z, xf, peak_threshold)
    export_peaks_txt([], [], peaks_z, SENSOR_TYPE, Z_CONTACT + "_clean", OUTPUT_DIR, peak_threshold)

    fig_fft, ax3_fft = plt.subplots(figsize=(15, 5), sharex=True)
    ax3_fft.plot(xf, amplitude_z, color='blue')
    ax3_fft.plot([f for f, _ in peaks_z], [a for _, a in peaks_z], "x", color='black', label="Peaks")
    ax3_fft.set_title('Z-axis Frequency Spectrum (Clean)')
    ax3_fft.set_ylabel('Amplitude')
    ax3_fft.set_xlabel('Frequency (Hz)')
    ax3_fft.grid(True)
    ax3_fft.legend()
    fft_output_filename = f'overall_frequency_domain_plot_{SENSOR_TYPE}_{Z_CONTACT}_clean_Z.png'
    fft_output_path = os.path.join(OUTPUT_DIR, fft_output_filename)
    plt.savefig(fft_output_path, dpi=300)
    plt.close(fig_fft)

# ===================== 靜止檔案分析 =====================
def static_file_analysis(settings):
    CLEAN_DIR = settings["clean_dir"]
    OUTPUT_DIR = settings["output_dir"]
    SENSOR_TYPE = settings["sensor_type"]
    Z_CONTACT = settings["z_contact"]
    SAMPLING_RATE = settings["sampling_rate"]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 取得所有 clean_X, clean_Y, clean_Z 檔案
    clean_x_files = sorted(
        glob.glob(os.path.join(CLEAN_DIR, "test_*_Raw_*_clean_X.csv")),
        key=lambda x: extract_raw_num(x)
    )
    clean_y_files = sorted(
        glob.glob(os.path.join(CLEAN_DIR, "test_*_Raw_*_clean_Y.csv")),
        key=lambda x: extract_raw_num(x)
    )
    clean_z_files = sorted(
        glob.glob(os.path.join(CLEAN_DIR, "test_*_Raw_*_clean_Z.csv")),
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
        print(f"在 {CLEAN_DIR} 中沒有找到完整的 X/Y/Z clean 檔案組。")
        return

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
    axs[0].plot(combined_x['t_s'], combined_x['raw'], color='navy', label='X raw', alpha=0.6)
    axs[0].plot(combined_x['t_s'], combined_x['clean'], color='blue', label='X clean')
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
    axs[2].plot(combined_z['t_s'], combined_z['raw'], color='grey', label='Z raw', alpha=0.6)
    axs[2].plot(combined_z['t_s'], combined_z['clean'], color='red', label='Z clean')
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
    axs_fft[0].plot(xf_x, amp_x, color='blue')
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
    axs_fft[2].plot(xf_z, amp_z, color='red')
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


# ===================== 主程式入口 =====================
if __name__ == "__main__":
    settings = get_user_settings()
    if settings["mode"] == "move":
        move_file_analysis(settings)
    elif settings["mode"] == "static":
        static_file_analysis(settings)
    print("所有分析與圖表繪製完成。")