
import os
import re
import glob
import argparse
import numpy as np
from scipy.signal import correlate
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import coherence


def extract_raw_num(filename):
    match = re.search(r'Raw_(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_vibration_data(vib_dir):
    """讀取震動檔案並合併"""
    files = sorted(
        glob.glob(os.path.join(vib_dir, f"{SENSOR_TYPE}_{Z_CONTACT}_*_vibration.csv")),
        key=lambda x: extract_raw_num(x)
    )

    if not files:
        print(f"沒有找到檔案。")
    else:
        print(f"找到 {len(files)} 個檔案。")

    vib_list = []
    for f in files:
        df = pd.read_csv(f)
        vib_list.append(df)
    #vib_df = pd.concat(vib_list, ignore_index=True)
    return vib_list

def load_optical_data(optical_dir):
    """讀取耦光檔案"""
    files = sorted(
        glob.glob(os.path.join(optical_dir, f"{SENSOR_TYPE}_{Z_CONTACT}_*_optical.csv")),
        key=lambda x: extract_raw_num(x)
    )
    if not files:
        print(f"沒有找到檔案。")
    else:
        print(f"找到 {len(files)} 個檔案。")

    opt_list = []
    for f in files:
        df = pd.read_csv(f)
        opt_list.append(df)
    #vib_df = pd.concat(vib_list, ignore_index=True)
    return opt_list

# --- 插值函式庫 ---
def interpolate_signal(x_old, y_old, x_new, method='linear'):
    """
    插值訊號
    x_old: 原始時間軸
    y_old: 原始訊號值
    x_new: 新的時間軸 (通常是震動的時間軸)
    method: 'linear', 'cubic', 'nearest' ...
    """
    # 使用 scipy.interpolate.interp1d 建立插值函式
    f_interp = interp1d(x_old, y_old, kind=method, fill_value="extrapolate")
    return f_interp(x_new)

# --- 交叉相關分析函式 ---
def cross_correlation_analysis(vibration_file, coupling_file, out_dir, interp_method='linear', fs=51200):
    """
    vibration_file: list of vibration DataFrames
    coupling_file: list of coupling DataFrames
    axis: 'X', 'Y', 'Z'
    interp_method: interpolation method ('linear', 'cubic', 'nearest' ...)
    fs: 取樣率 (Hz)，預設 51200
    """
    corr_dir = os.path.join(out_dir, "correlation")
    cohe_dir = os.path.join(out_dir, "cheherence")
    os.makedirs(corr_dir, exist_ok=True)
    os.makedirs(cohe_dir, exist_ok=True)


    max_corr = []
    lag_at_max = []
    segment = 0
    for vib, coup in zip(vibration_file, coupling_file):
        print(f"Processing segment {segment}")
        segment += 1
        for axis in ['X', 'Y', 'Z']:
            print(f"  Analyzing {axis}-axis")
            # 震動數據
            t_vib = vib["X_Time"].values
            vib_axis = vib[f"{axis}_Amp"].values

            # 耦光數據
            t_coup = coup["Time_shifted"].values
            power = coup["Optical_Power"].values

            # --- 插值耦光訊號到震動時間軸 ---
            power_interp = interpolate_signal(t_coup, power, t_vib, method=interp_method)

            # 正規化
            vib_norm = (vib_axis - np.mean(vib_axis)) / np.std(vib_axis)
            power_norm = (power_interp - np.mean(power_interp)) / np.std(power_interp)

            # 計算交叉相關 (normalized)
            corr = correlate(vib_norm, power_norm, mode='full')
            corr = corr / len(vib_norm)   # 轉成相關係數 [-1, 1]

            # 計算 lag (轉換成秒)
            lags = np.arange(-len(vib_norm)+1, len(power_norm)) / fs

            # 找最大相關性
            max_corr.append(np.max(np.abs(corr)))
            lag_at_max.append(lags[np.argmax(np.abs(corr))])

            # 繪圖
            plt.figure(figsize=(8,4))
            plt.plot(lags, corr)
            plt.title(f'Cross-correlation on {SENSOR_TYPE}_{Z_CONTACT} ({axis}-axis vs Optical Power, method={interp_method})')
            plt.xlabel('Lag (seconds)')
            plt.ylabel('Correlation coefficient')
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.savefig(os.path.join(corr_dir, f'cross_correlation_{SENSOR_TYPE}_{Z_CONTACT}_{segment-1}_{axis}.png'))
            plt.close()

            # ---- 相干性分析 ----
            nperseg = 2048  # 可調整
            f, Cxy = coherence(vib_axis, power_interp, fs=fs, nperseg=nperseg)

            plt.figure(figsize=(8,4))
            plt.semilogy(f, Cxy)
            plt.title(f'Coherence {SENSOR_TYPE}_{Z_CONTACT} ({axis}-axis vs Optical Power)')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Coherence')
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(cohe_dir, f'coherence_{SENSOR_TYPE}_{Z_CONTACT}_{segment-1}_{axis}.png'))
            plt.close()
        
    return max_corr, lag_at_max

def main():
    parser = argparse.ArgumentParser(description="對齊震動與耦光數據並輸出圖與Excel")
    parser.add_argument("--vib_dir", type=str, required=True, help="震動數據資料夾路徑")
    parser.add_argument("--opt_dir", type=str, required=True, help="耦光數據資料夾路徑")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出結果資料夾")
    parser.add_argument("--interval", type=int, default=2, help="對齊時間區段 (秒)，預設 2 秒")
    parser.add_argument("--sensor", type=str, required=True, help="sensor type: Probe/6Axis")
    parser.add_argument("--zcontact", type=str, required=True, help="stop/500ms/1500ms")
    args = parser.parse_args()

    global SENSOR_TYPE, Z_CONTACT
    SENSOR_TYPE = args.sensor
    Z_CONTACT = args.zcontact
    
    vib_df = load_vibration_data(args.vib_dir)
    print(f"Loaded {len(vib_df)} vibration segments.")
    optical_df = load_optical_data(args.opt_dir)
    #optical_df = pd.read_csv(args.vib_dir)
    print(f"Loaded optical data with {len(optical_df)} records.")

    max_corr, lag = cross_correlation_analysis(vib_df, optical_df, args.output_dir, 'linear')

    for i, (mc, lg) in enumerate(zip(max_corr, lag)):
        print(f"Segment {i}: Max Correlation = {mc:.4f}, Lag at Max = {lg:.6f} seconds")
    
if __name__ == "__main__":
    main()
