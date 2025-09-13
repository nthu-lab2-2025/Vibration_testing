"""
#python build_stop_baseline.py --data_dir "D:\專題\矽光子\Data\20250807\sensor_on_Probe\Probe_Stop\20250807170238" --analysis_dir "D:\專題\矽光子\Data\20250807\sensor_on_Probe\Probe_Stop\test"

"""

import argparse

def main():
    parser = argparse.ArgumentParser(description='建立靜止基準')
    parser.add_argument('--data_dir', required=True, help='資料來源資料夾')
    parser.add_argument('--analysis_dir', required=True, help='分析輸出資料夾')
    args = parser.parse_args()
    DATA_DIR = args.data_dir
    ANALYSIS_DIR = args.analysis_dir
    print(f'[INFO] 資料來源: {DATA_DIR}')
    print(f'[INFO] 輸出資料夾: {ANALYSIS_DIR}')

    """
    目標：從「chuck 靜止」的原始震動資料，建立一份三軸（X、Y、Z）在靜止狀態下的統計基準。
    根據 sensor_on_Probe / Probe_stop的資料

    """
    import os
    import json
    import glob
    import numpy as np
    import pandas as pd
    from scipy.signal import welch
    from scipy.stats import kurtosis
    from scipy.integrate import trapezoid 
    import matplotlib.pyplot as plt

    # === 路徑設定 ===
    # DATA_DIR = r"D:\專題\矽光子\Data\20250807\sensor_on_Probe\Probe_Stop\20250807170238"  # 輸入不變
    # ANALYSIS_DIR = r"D:\專題\矽光子\Data\20250807\sensor_on_Probe\Probe_Stop\analysis"    # 輸出集中到這裡
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    FIG_DIR = ANALYSIS_DIR 

    #  用時間欄位估計取樣率（Hz）
    def estimate_fs(t): # t 為時間欄位 (X_Time(S))
        dt = np.diff(t) # 計算相鄰的時間差
        dt = dt[(dt > 0) & np.isfinite(dt)]  # 過濾掉非正數和非有限值
        # 先取中位數，再取倒數得到估計取樣率(HZ) 
        # 如果 dt 為空，則返回 NaN
        return 1.0 / np.median(dt) if len(dt) else np.nan

    # 計算用來判定基準的features
    def time_features(sig):
        rms = np.sqrt(np.mean(sig**2)) # rms均方根值 sig 為某一軸的震動訊號
        peak = np.max(np.abs(sig)) # 最大值
        mean = np.mean(sig) # 平均值
        std = np.std(sig) # 標準差
        crest = peak / rms if rms > 0 else np.inf # 峰均比
        kurt = kurtosis(sig, fisher=False)  # kurt 峰度 (Pearson 定義，常態=3
        return dict(rms=rms, peak=peak, mean=mean, std=std, crest_factor=crest, kurtosis=kurt)

    # 計算頻帶能量
    def band_energy(sig, fs, f_low, f_high):
        # 如果取樣率無效或小於等於0，則返回 NaN
        if not np.isfinite(fs) or fs <= 0:
            return np.nan
        nperseg = min(4096, max(256, len(sig)//4))
        #  welch()：把時域訊號換成「各頻率的能量密度」
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        # 建立頻帶遮罩，只保留 [f_low, f_high] 之內的頻點
        mask = (freqs >= f_low) & (freqs <= f_high)
        if not np.any(mask):
            return 0.0
        #  改用 scipy.integrate.trapezoid（取代 np.trapz）
        return trapezoid(psd[mask], freqs[mask]) # 把該頻帶的功率密度累加成頻帶能量

    # 確保輸出目錄存在# 欄位 mapping：容忍大小寫、括號、底線差異
    def pick_col(df, candidates):
        norm = {str(c).lower().replace("_","").replace("(","").replace(")",""): c for c in df.columns}
        for cand in candidates:
            key = cand.lower().replace("_","").replace("(","").replace(")","")
            if key in norm:
                return norm[key]
        raise KeyError(f"找不到欄位，候選：{candidates}")
    
    # 處理單個檔案，計算各軸的特徵
    # 單檔處理（逐軸容錯；找不到某軸就只略過該軸）
    def process_file(path):
        df = pd.read_csv(path, sep=None, engine="python")
        feats = {}
        def get_axis(axis, t_cands, s_cands):
            try:
                t = df[pick_col(df, t_cands)].to_numpy()
                s = df[pick_col(df, s_cands)].to_numpy()
                fs = estimate_fs(t)
                tf = time_features(s)
                feats[axis] = {
                    **tf,
                    "fs_est": fs,
                    "band_energy_10_200":  band_energy(s, fs, 10, 200),
                    "band_energy_200_2000": band_energy(s, fs, 200, 2000),
                    "n_samples": len(s),
                    "duration_s": (len(s) - 1) / fs if np.isfinite(fs) and fs > 0 else np.nan,
                }
            except Exception as e:
                print(f"[axis-skip] {os.path.basename(path)} {axis}: {e}")
                
        # 嘗試獲取X、Y、Z三軸數據
        get_axis("X", 
                ["X_Time(S)","X Time(S)","X_Time","x_time(s)","XTimeS","X_Time_S"],
                ["X_Sensor_Amplitude(V)","X Amplitude","X_Value","x_sensor_amplitude(v)","XAmplitude","X_Value(V)"])
        get_axis("Y", 
                ["Y_Time(S)","Y_Time","y_time(s)","YTimeS","Y_Time_S"],
                ["Y_Sensor_Amplitude(V)","Y Amplitude","Y_Value","y_sensor_amplitude(v)","YAmplitude","Y_Value(V)"])
        get_axis("Z", 
                ["Z_Time(S)","Z_Time","z_time(s)","ZTimeS","Z_Time_S"],
                ["Z_Sensor_Amplitude(V)","Z Amplitude","Z_Value","z_sensor_amplitude(v)","ZAmplitude","Z_Value(V)"])
        
        if not feats:
            raise KeyError("三軸都找不到合法欄位")
        return feats
    

    # 計算加權平均值
    def weighted_mean(series, weights):
        s, w = series.to_numpy(float), weights.to_numpy(float)
        mask = np.isfinite(s) & np.isfinite(w)
        return np.average(s[mask], weights=w[mask]) if np.any(mask) else np.nan



    # === 讀取設定檔 ===
    files = sorted(glob.glob(os.path.join(DATA_DIR, "test_*_Raw_*.csv")))
    if not files:
        raise FileNotFoundError("找不到符合的檔案")

    # 逐檔處理，計算各軸的特徵
    print(f"找到 {len(files)} 個檔案，開始處理...")
    per_file_rows = []
    for f in files:
        feats = process_file(f)
        row = {"file": os.path.basename(f)}
        for axis in ["X", "Y", "Z"]:
            for k, v in feats[axis].items():
                row[f"{axis}_{k}"] = v
        per_file_rows.append(row)

    per_file_df = pd.DataFrame(per_file_rows)

    # 計算加權平均值
    baseline = {"axes": {}}
    for axis in ["X", "Y", "Z"]:
        w = per_file_df[f"{axis}_n_samples"].fillna(0)
        axis_summary = {
            "fs_est": weighted_mean(per_file_df[f"{axis}_fs_est"], w),

            "rms_mean": weighted_mean(per_file_df[f"{axis}_rms"], w),
            "rms_std": per_file_df[f"{axis}_rms"].std(ddof=0),

            "peak_mean": weighted_mean(per_file_df[f"{axis}_peak"], w),
            "peak_std": per_file_df[f"{axis}_peak"].std(ddof=0),

            "mean_mean": weighted_mean(per_file_df[f"{axis}_mean"], w),
            "mean_std": per_file_df[f"{axis}_mean"].std(ddof=0),

            "std_mean": weighted_mean(per_file_df[f"{axis}_std"], w),
            "std_std": per_file_df[f"{axis}_std"].std(ddof=0),

            "crest_factor_mean": weighted_mean(per_file_df[f"{axis}_crest_factor"], w),
            "crest_factor_std": per_file_df[f"{axis}_crest_factor"].std(ddof=0),

            "kurtosis_mean": weighted_mean(per_file_df[f"{axis}_kurtosis"], w),
            "kurtosis_std": per_file_df[f"{axis}_kurtosis"].std(ddof=0),

            "band_energy_10_200_mean": weighted_mean(per_file_df[f"{axis}_band_energy_10_200"], w),
            "band_energy_10_200_std": per_file_df[f"{axis}_band_energy_10_200"].std(ddof=0),

            "band_energy_200_2000_mean": weighted_mean(per_file_df[f"{axis}_band_energy_200_2000"], w),
            "band_energy_200_2000_std": per_file_df[f"{axis}_band_energy_200_2000"].std(ddof=0),
        }
        baseline["axes"][axis] = axis_summary

    baseline["files_count"] = len(per_file_df)
    baseline["notes"] = "以 Probe_Stop 靜止資料建立的三軸震動基準"

    print("[debug] writing baseline axes keys:", list(baseline["axes"].keys()))

    # 輸出結果
    print("計算完成，輸出結果...")
    out_json = os.path.join(ANALYSIS_DIR, "baseline_summary.json")
    out_csv = os.path.join(ANALYSIS_DIR, "per_file_metrics.csv")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    per_file_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"已輸出：\n  {out_json}\n  {out_csv}")


    # ====== 圖像化（也輸出到 analysis 目錄） ======
    """
    可調參數:
    1. PSD x 軸範圍：plt.xlim(0, 3000) 可改，例如 5000。
    2. Welch 切片長度：nperseg 的上下限（256 與 4096）可調。
    3. 時間波形抽樣數：N // 2000 改成 N // 4000 會更細。
    4. 挑圖的檔案：indices=(0,30,60) 改成任何你想看的索引。
    5. RMS 長條圖：如果想看排序後的分布，可在 vals 先排序再畫
    """
    # RMS 長條圖
    def plot_rms_bars(per_file_df, axis):
        vals = per_file_df[f"{axis}_rms"].to_numpy()
        idx = np.arange(len(vals))
        plt.figure()
        plt.bar(idx, vals)
        plt.xlabel("file index")
        plt.ylabel(f"{axis} RMS (V)")
        plt.title(f"{axis}-axis RMS across files (static baseline)")
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"{axis}_rms_bars.png")
        plt.savefig(out, dpi=160)
        plt.close()

    for ax in ["X", "Y", "Z"]:
        plot_rms_bars(per_file_df, ax)

    # 頻譜（PSD）平均曲線＋標準差陰影
    def compute_psd(sig, fs):
        nperseg = min(4096, max(256, len(sig)//4))
        f, p = welch(sig, fs=fs, nperseg=nperseg)
        return f, p

    def plot_mean_psd(files, axis):
        psd_list = []
        f_ref = None
        for fpath in files:
            df = pd.read_csv(fpath)
            if axis == "X":
                t = df["X_Time(S)"].to_numpy()
                s = df["X_Sensor_Amplitude(V)"].to_numpy()
            elif axis == "Y":
                t = df["Y_Time(S)"].to_numpy()
                s = df["Y_Sensor_Amplitude(V)"].to_numpy()
            else:
                t = df["Z_Time(S)"].to_numpy()
                s = df["Z_Sensor_Amplitude(V)"].to_numpy()

            dt = np.diff(t)
            dt = dt[(dt > 0) & np.isfinite(dt)]
            if len(dt) == 0:
                continue
            fs = 1.0 / np.median(dt)
            f, p = compute_psd(s, fs)

            if f_ref is None:
                f_ref = f
            else:
                n = min(len(f_ref), len(f))
                f_ref = f_ref[:n]
                p = p[:n]
            psd_list.append(p)

        if len(psd_list) == 0 or f_ref is None:
            return

        psd_arr = np.vstack(psd_list)
        m = psd_arr.mean(axis=0)
        s = psd_arr.std(axis=0)

        plt.figure()
        plt.semilogy(f_ref, m) # y 軸對數，頻譜差異更容易看
        plt.fill_between(f_ref, np.maximum(m - s, 1e-20), m + s, alpha=0.3) # 出 平均 ±1σ 的陰影區，表示正常波動範圍
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD")
        plt.title(f"{axis}-axis PSD (mean ±1σ, static baseline)")
        plt.xlim(0, 3000)  # 可視需要調整
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"{axis}_psd_mean_std.png")
        plt.savefig(out, dpi=160)
        plt.close()

    plot_mean_psd(files, "X")
    plot_mean_psd(files, "Y")
    plot_mean_psd(files, "Z")

    # 低頻 vs 高頻能量散點圖
    def plot_low_high_scatter(per_file_df, axis):
        # 獲取低頻和高頻能量
        low = per_file_df[f"{axis}_band_energy_10_200"].to_numpy()
        high = per_file_df[f"{axis}_band_energy_200_2000"].to_numpy()
        # 每一點是一個檔案  x=低頻能量、y=高頻能量
        plt.figure()
        plt.scatter(low, high, s=16)
        plt.xlabel(f"{axis} band energy 10–200 Hz")
        plt.ylabel(f"{axis} band energy 200–2000 Hz")
        plt.title(f"{axis}-axis band energy scatter (static baseline)")
        plt.tight_layout()
        out = os.path.join(FIG_DIR, f"{axis}_band_energy_scatter.png")
        plt.savefig(out, dpi=160)
        plt.close()

    for ax in ["X", "Y", "Z"]:
        plot_low_high_scatter(per_file_df, ax)

    def quick_plot_time_series(files, indices=(0, 30, 60)):
        for idx in indices:
            if idx < 0 or idx >= len(files):
                continue
            f = files[idx]
            df = pd.read_csv(f)
            for axis in ["X", "Y", "Z"]:
                if axis == "X":
                    t = df["X_Time(S)"].to_numpy()
                    s = df["X_Sensor_Amplitude(V)"].to_numpy()
                elif axis == "Y":
                    t = df["Y_Time(S)"].to_numpy()
                    s = df["Y_Sensor_Amplitude(V)"].to_numpy()
                else:
                    t = df["Z_Time(S)"].to_numpy()
                    s = df["Z_Sensor_Amplitude(V)"].to_numpy()

                # 降採樣以便繪圖
                N = len(s)
                step = max(1, N // 2000)
                t_ds = t[::step]
                s_ds = s[::step]

                plt.figure()
                plt.plot(t_ds, s_ds)
                plt.xlabel("Time (s)")
                plt.ylabel(f"{axis} Amplitude (V)")
                plt.title(f"Time series ({axis}) - file #{idx}: {os.path.basename(f)}")
                plt.tight_layout()
                out = os.path.join(FIG_DIR, f"time_{axis}_file{idx}.png")
                plt.savefig(out, dpi=160)
                plt.close()

    quick_plot_time_series(files)
    print(f"[圖像化] 已輸出到：{FIG_DIR}")

if __name__ == '__main__':
    main()
