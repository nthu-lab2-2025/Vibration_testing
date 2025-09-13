'''加入cross-correlation'''

# step0_align_plot.py
# -*- coding: utf-8 -*-
import os, re, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# --------- 讀檔與工具 ---------
def extract_raw_num(filename):
    m = re.search(r'Raw_(\d+)', filename)
    return int(m.group(1)) if m else -1

def load_vibration_list(vib_dir):
    """回傳 list[DataFrame]，每個檔一個 df（約2秒），並統一欄位名。"""
    files = sorted(glob.glob(os.path.join(vib_dir, "test_*_Raw_*.csv")),
                   key=lambda x: extract_raw_num(x))
    if not files:
        raise FileNotFoundError(f"沒有找到檔案：{vib_dir}\\test_*_Raw_*.csv")
    print(f"[VIB] 找到 {len(files)} 個檔案。")

    lst = []
    for f in files:
        df = pd.read_csv(f)
        df = df.rename(columns={
            "X_Time(S)": "X_Time", "X_Sensor_Amplitude(V)": "X_Amp",
            "Y_Time(S)": "Y_Time", "Y_Sensor_Amplitude(V)": "Y_Amp",
            "Z_Time(S)": "Z_Time", "Z_Sensor_Amplitude(V)": "Z_Amp"
        })
        for need in ["X_Time","Y_Time","Z_Time","X_Amp","Y_Amp","Z_Amp"]:
            if need not in df.columns:
                raise KeyError(f"{os.path.basename(f)} 缺少欄位 {need}，實際欄位：{list(df.columns)}")
        lst.append(df)
    return lst

def load_optical(opt_file):
    df = pd.read_csv(opt_file)
    df = df.rename(columns={"Time(ms)":"Time_ms","Optical_Power(dBm)":"Optical_Power"})
    if "Time_ms" not in df.columns or "Optical_Power" not in df.columns:
        raise KeyError(f"{opt_file} 需含 'Time(ms)' 與 'Optical_Power(dBm)' 欄位")
    df["Time_s"] = df["Time_ms"] / 1000.0
    return df

def est_fs_from_t(t):
    dt = np.diff(t)
    dt = dt[(dt>0) & np.isfinite(dt)]
    return 1.0/np.median(dt) if dt.size else np.nan

def zscore(x):
    m, s = np.mean(x), np.std(x)
    return (x-m) / (s + 1e-12)

def bandpass(sig, fs, lo=1.0, hi=200.0, order=4):
    ny = fs * 0.5
    hi = min(hi, ny*0.99)
    if hi <= 0 or hi <= lo:
        return sig
    b, a = signal.butter(order, [lo/ny, hi/ny], btype='band')
    return signal.filtfilt(b, a, sig)

# --------- 核心：把震動接成一條 + 對齊 + 裁交集 ---------
def concat_vib_to_global(vib_list, interval=2.0):
    rows = []
    for i, df in enumerate(vib_list):
        t0 = i * interval
        tx = df["X_Time"].to_numpy()
        tx = tx - tx.min() + t0  # 每檔歸零後加起點
        rows.append(pd.DataFrame({
            "t": tx,
            "X_Amp": df["X_Amp"].to_numpy(),
            "Y_Amp": df["Y_Amp"].to_numpy(),
            "Z_Amp": df["Z_Amp"].to_numpy()
        }))
    vib_all = pd.concat(rows, ignore_index=True)
    fs_v = est_fs_from_t(vib_all["t"].to_numpy())
    return vib_all, fs_v

def global_align_and_crop(vib_all, fs_vib, optical_df, max_shift_s=3.0, bp_lo=1.0, bp_hi=200.0):
    # 參考：三軸能量包絡
    v_env = np.sqrt(vib_all["X_Amp"]**2 + vib_all["Y_Amp"]**2 + vib_all["Z_Amp"]**2).to_numpy()
    t_v = vib_all["t"].to_numpy()
    v_env = bandpass(v_env - np.mean(v_env), fs_vib, bp_lo, bp_hi)

    t_o = optical_df["Time_s"].to_numpy()
    y   = optical_df["Optical_Power"].to_numpy()
    # 光學也帶通以強化共同成分
    fs_o = est_fs_from_t(t_o) if np.isfinite(est_fs_from_t(t_o)) else fs_vib
    y_bp = bandpass(y - np.mean(y), fs_o, bp_lo, bp_hi)

    # 先把 v_env 插到光學時間軸上，再做交叉相關找 offset
    v_on_o = np.interp(t_o, t_v, v_env)
    y0 = zscore(y_bp); x0 = zscore(v_on_o)
    full = np.correlate(y0, x0, mode='full')              # lags: -(n-1)~(n-1)
    lags = np.arange(-len(y0)+1, len(y0), dtype=int)
    max_lag = int(round(max(0.0, max_shift_s) * fs_o))
    m = (lags >= -max_lag) & (lags <= max_lag)
    if not np.any(m): m = slice(None)
    best_lag = int(lags[m][np.argmax(full[m])])
    offset_s = best_lag / fs_o            # lag>0 表 y 落後，要往右移 y

    # 套用 offset，裁出與震動重疊區
    t_o_shift = t_o + max(0.0, offset_s)
    t_start = max(t_v.min(), t_o_shift.min())
    t_end   = min(t_v.max(), t_o_shift.max())
    keep = (t_o_shift >= t_start) & (t_o_shift <= t_end)
    opt_crop = optical_df.loc[keep].copy()
    opt_crop["Time_s_aligned"] = t_o_shift[keep]

    print(f"[ALIGN] best offset = {offset_s:.3f}s; overlap = {t_start:.3f} ~ {t_end:.3f}s "
          f"(vib≈{t_v.min():.3f}~{t_v.max():.3f}, opt≈{t_o_shift.min():.3f}~{t_o_shift.max():.3f})")
    return opt_crop, (t_start, t_end), offset_s

# --------- 畫圖：整段 + 每2秒 ---------
def plot_full(vib_all, opt_crop, out_png, interval):
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    t = vib_all["t"].to_numpy()
    axes[0].plot(t, vib_all["X_Amp"].to_numpy(), label="X")
    axes[1].plot(t, vib_all["Y_Amp"].to_numpy(), label="Y")
    axes[2].plot(t, vib_all["Z_Amp"].to_numpy(), label="Z")
    axes[3].plot(opt_crop["Time_s_aligned"], opt_crop["Optical_Power"], label="Optical")

    for ax in axes:
        ax.legend(loc="upper right")
        ax.set_ylabel("Amp")

    # 每2秒一條直線
    t0, t1 = t.min(), t.max()
    k = int(np.ceil((t1 - t0) / interval))
    for i in range(k+1):
        x = t0 + i*interval
        for ax in axes:
            ax.axvline(x, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Vibration (X/Y/Z) & Optical (aligned) — full length with 2s markers")
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {out_png}")

def plot_windows_and_save(vib_list, opt_crop, out_dir, sensor, zcontact, interval):
    plot_dir = os.path.join(out_dir, "plots")
    csv_dir  = os.path.join(out_dir, "csv")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # 光學用對齊後時間取窗
    for seg_idx, vib_seg in enumerate(vib_list):
        t_start = seg_idx * interval
        t_end   = (seg_idx + 1) * interval

        # 震動片段（原始每檔的 0~2 秒）
        x_t, y_t, z_t = vib_seg["X_Time"], vib_seg["Y_Time"], vib_seg["Z_Time"]
        x_a, y_a, z_a = vib_seg["X_Amp"],  vib_seg["Y_Amp"],  vib_seg["Z_Amp"]

        # 光學片段（對齊後時間）
        m = (opt_crop["Time_s_aligned"] >= t_start) & (opt_crop["Time_s_aligned"] < t_end)
        opt_seg = opt_crop.loc[m]
        if opt_seg.empty:
            continue
        # 將光學時間改為視窗內相對秒數便於疊圖
        opt_seg = opt_seg.copy()
        opt_seg["Time_win"] = opt_seg["Time_s_aligned"] - t_start

        # 畫單窗
        fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=False)
        axes[0].plot(x_t, x_a, label="X", color="r"); axes[0].legend(loc="upper right")
        axes[1].plot(y_t, y_a, label="Y", color="g"); axes[1].legend(loc="upper right")
        axes[2].plot(z_t, z_a, label="Z", color="b"); axes[2].legend(loc="upper right")
        axes[3].plot(opt_seg["Time_win"], opt_seg["Optical_Power"], label="Optical", color="orange")
        axes[0].set_ylabel("X(V)"); axes[1].set_ylabel("Y(V)"); axes[2].set_ylabel("Z(V)")
        axes[3].set_ylabel("Opt(dBm)"); axes[3].set_xlabel("Time in window (s)")
        fig.suptitle(f"{sensor}_{zcontact}  seg#{seg_idx}  [{t_start:.2f}~{t_end:.2f}s]")
        fig.tight_layout(rect=[0,0.04,1,0.95])
        seg_png = os.path.join(plot_dir, f"{sensor}_{zcontact}_{seg_idx:04d}.png")
        fig.savefig(seg_png, dpi=140); plt.close(fig)
        print(f"[PLOT] {seg_png}")

        # 存窗 CSV（後續做特徵用）
        vib_out = vib_seg.copy()
        vib_out["Interval_Start"] = t_start; vib_out["Interval_End"] = t_end
        opt_out = opt_seg[["Time_s_aligned","Time_win","Optical_Power"]].copy()
        opt_out["Interval_Start"] = t_start; opt_out["Interval_End"] = t_end

        vib_out.to_csv(os.path.join(csv_dir, f"{sensor}_{zcontact}_{seg_idx:04d}_vibration.csv"), index=False)
        opt_out.to_csv(os.path.join(csv_dir, f"{sensor}_{zcontact}_{seg_idx:04d}_optical.csv"), index=False)

# --------- 主流程 ---------
def main():
    ap = argparse.ArgumentParser(description="Step0：先對齊→裁交集→畫整段&每2秒一圖")
    ap.add_argument("--vib_dir",    required=True, help="震動資料夾（test_*_Raw_*.csv）")
    ap.add_argument("--opt_file",   required=True, help="耦光單一CSV（Time(ms), Optical_Power(dBm)）")
    ap.add_argument("--output_dir", required=True, help="輸出資料夾")
    ap.add_argument("--interval",   type=int, default=2, help="每窗秒數，預設 2")
    ap.add_argument("--sensor",     required=True, help="Probe / 6Axis（僅做檔名標示）")
    ap.add_argument("--zcontact",   required=True, help="stop / 500ms / 1500ms（檔名標示）")
    ap.add_argument("--max_shift_s",type=float, default=3.0, help="交叉相關搜尋延遲最大秒數")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    vib_list = load_vibration_list(args.vib_dir)
    optical  = load_optical(args.opt_file)

    # 1) 震動接成一條
    vib_all, fs_v = concat_vib_to_global(vib_list, interval=args.interval)
    print(f"[VIB] fs ≈ {fs_v:.1f} Hz, 長度 ≈ {vib_all['t'].max():.3f}s")

    # 2) 全域對齊 + 裁交集
    opt_crop, (ovl_start, ovl_end), offset_s = global_align_and_crop(
        vib_all, fs_v, optical, max_shift_s=args.max_shift_s, bp_lo=1.0, bp_hi=200.0
    )

    # 3) 整段總覽圖（含每2秒直線）
    plot_full(vib_all, opt_crop, os.path.join(args.output_dir, f"{args.sensor}_{args.zcontact}_full.png"), args.interval)

    # 4) 每2秒一圖 + 存逐窗CSV
    plot_windows_and_save(vib_list, opt_crop, args.output_dir, args.sensor, args.zcontact, args.interval)

    # 5) 紀錄對齊摘要
    pd.DataFrame([{
        "offset_seconds": offset_s,
        "overlap_start": ovl_start,
        "overlap_end": ovl_end,
        "vib_total_s": float(vib_all["t"].max()),
        "opt_total_s": float(optical["Time_s"].max())
    }]).to_csv(os.path.join(args.output_dir, "align_summary.csv"), index=False)
    print(f"[SAVE] {os.path.join(args.output_dir, 'align_summary.csv')}")

if __name__ == "__main__":
    main()
