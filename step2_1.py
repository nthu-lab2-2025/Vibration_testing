#先用 nperseg = 2048（你的程式預設），頻率解析度大約 
#𝑓𝑠/2048≈25𝐻𝑧

#如果想看低頻細節（例如 <10 Hz），建議把 nperseg 調大到 8192 或 16384。

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import correlate, coherence


def extract_raw_num(filename):
    match = re.search(r'Raw_(\d+)', filename)
    return int(match.group(1)) if match else -1


def load_vibration_data(vib_dir):
    """讀取所有震動檔案並合併成單一 DataFrame，時間軸改成 Time_shifted"""
    files = sorted(
        glob.glob(os.path.join(vib_dir, "test_*_Raw_*.csv")),
        key=lambda x: extract_raw_num(x)
    )

    if not files:
        raise FileNotFoundError("沒有找到震動檔案")

    vib_list = []
    for i, f in enumerate(files, start=1):
        df = pd.read_csv(f)
        # 在原始時間欄位上加偏移，得到全域時間
        df["Time_shifted"] = 2 * (i - 1) + df["X_Time(S)"]
        vib_list.append(df)

    # 合併
    vib_all = pd.concat(vib_list, ignore_index=True)
    return vib_all


def load_optical_data(optical_file):
    """讀取單一耦光檔案"""
    return pd.read_csv(optical_file)


def interpolate_signal(x_old, y_old, x_new, method='linear'):
    """插值訊號"""
    f_interp = interp1d(x_old, y_old, kind=method, fill_value="extrapolate")
    return f_interp(x_new)


def cross_corr_and_coherence(vib_all, coup, out_dir, fs=51200, interp_method='linear'):
    os.makedirs(out_dir, exist_ok=True)

    # 光學數據
    t_coup = coup["Time(ms)"].values / 1000  # 轉成秒
    power = coup["Optical_Power(dBm)"].values

    # 插值光功率到震動時間軸
    t_vib = vib_all["Time_shifted"].values
    power_interp = interpolate_signal(t_coup, power, t_vib, method=interp_method)

    results = {}

    for axis in ['X', 'Y', 'Z']:
        vib_axis = vib_all[f"{axis}_Sensor_Amplitude(V)"].values

        min_len = min(len(vib_axis), len(power_interp))
        vib_axis = vib_axis[:min_len]
        power_interp = power_interp[:min_len]

        # ----------- 交叉相關 -----------
        vib_norm = (vib_axis - np.mean(vib_axis)) / np.std(vib_axis)
        power_norm = (power_interp - np.mean(power_interp)) / np.std(power_interp)

        corr = correlate(vib_norm, power_norm, mode='full')
        corr = corr / len(vib_norm)
        lags = np.arange(-len(vib_norm)+1, len(power_norm)) / fs

        max_corr = np.max(np.abs(corr))
        lag_at_max = lags[np.argmax(np.abs(corr))]
        results[axis] = (max_corr, lag_at_max)

        # ----------- 相干性 -----------
        nperseg = 8192  # 可調整，影響頻率解析度
        f, Cxy = coherence(vib_axis, power_interp, fs=fs, nperseg=nperseg)

        # ----------- 繪圖 (相關 + 相干性放一起) -----------
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # 交叉相關圖
        axs[0].plot(lags, corr)
        axs[0].set_title(f'Cross-correlation ({axis}-axis vs Optical Power)')
        axs[0].set_xlabel('Lag [s]')
        axs[0].set_ylabel('Correlation coefficient')
        axs[0].grid(True, linestyle="--", alpha=0.6)

        # 相干性圖
        axs[1].semilogy(f, Cxy)
        axs[1].set_title(f'Coherence ({axis}-axis vs Optical Power)')
        axs[1].set_xlabel('Frequency [Hz]')
        axs[1].set_ylabel('Coherence')
        axs[1].grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'corr_coherence_{SENSOR_TYPE}_{Z_CONTACT}_{axis}.png'))
        plt.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="合併震動檔案並進行交叉相關與相干性分析")
    parser.add_argument("--vib_dir", type=str, required=True, help="震動數據資料夾路徑")
    parser.add_argument("--opt_file", type=str, required=True, help="單一耦光數據檔案路徑")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出結果資料夾")
    parser.add_argument("--sensor", type=str, required=True, help="sensor type: Probe/6Axis")
    parser.add_argument("--zcontact", type=str, required=True, help="stop/500ms/1500ms")
    args = parser.parse_args()

    global SENSOR_TYPE, Z_CONTACT
    SENSOR_TYPE = args.sensor
    Z_CONTACT = args.zcontact

    vib_all = load_vibration_data(args.vib_dir)
    print(f"合併後震動資料長度: {len(vib_all)} samples, 總時間: {vib_all['Time_shifted'].iloc[-1]:.2f} 秒")

    optical_df = load_optical_data(args.opt_file)
    print(f"耦光資料長度: {len(optical_df)} records")

    results = cross_corr_and_coherence(vib_all, optical_df, args.output_dir)

    # 輸出到文字檔
    out_file = os.path.join(args.output_dir, f"correlation_results_{SENSOR_TYPE}_{Z_CONTACT}.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        for axis, (mc, lg) in results.items():
            line = f"{axis}-axis: Max Correlation = {mc:.4f}, Lag at Max = {lg:.6f} 秒\n"
            f.write(line)
            #print(line.strip())  # 同時也在螢幕顯示



if __name__ == "__main__":
    main()
