import os
import re
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def extract_raw_num(filename):
    match = re.search(r'Raw_(\d+)', filename)
    return int(match.group(1)) if match else -1

def load_vibration_data(vib_dir):
    """讀取震動檔案並合併"""
    files = sorted(
        glob.glob(os.path.join(vib_dir, "test_*_Raw_*.csv")),
        key=lambda x: extract_raw_num(x)
    )

    if not files:
        print(f"沒有找到檔案。")
    else:
        print(f"找到 {len(files)} 個檔案。")

    vib_list = []
    for f in files:
        df = pd.read_csv(f)
        df = df.rename(columns={
            "X_Time(S)": "X_Time", "X_Sensor_Amplitude(V)": "X_Amp",
            "Y_Time(S)": "Y_Time", "Y_Sensor_Amplitude(V)": "Y_Amp",
            "Z_Time(S)": "Z_Time", "Z_Sensor_Amplitude(V)": "Z_Amp"
        })
        vib_list.append(df)
    #vib_df = pd.concat(vib_list, ignore_index=True)
    return vib_list

def load_optical_data(optical_file):
    """讀取耦光檔案"""
    df = pd.read_csv(optical_file)
    df = df.rename(columns={"Time(ms)": "Time_ms", "Optical_Power(dBm)": "Optical_Power"})
    df["Time_s"] = df["Time_ms"] / 1000.0  # 轉換成秒
    return df

def align_and_plot(vib_list, optical_df, out_dir, interval=2):
    """每 interval 秒對齊數據並畫圖與存檔"""
    plot_dir = os.path.join(out_dir, "plots")
    data_dir = os.path.join(out_dir, "csv")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # 時間範圍
    start_time = max(vib_list[0]["X_Time"].min(), optical_df["Time_s"].min())
    end_time = min(len(vib_list)*2, optical_df["Time_s"].max())
    print(f"start_time: {start_time}, end_time: {end_time}")

    t = start_time
    segment = 0
    while t < end_time:
        #t_start, t_end = t, t + interval
        t_start = t
        t_end = interval*(segment + 1)
        vib_seg = vib_list[segment]
        print(f"Processing segment {segment}: {t_start} to {t_end}")
        #vib_seg = vib_df[(vib_df["X_Time"] >= t_start) & (vib_df["X_Time"] < t_end)]
        optical_seg = optical_df.loc[(optical_df["Time_s"] >= t_start) & (optical_df["Time_s"] < t_end)].copy()
        optical_seg["Time_shifted"] = optical_seg["Time_s"] - t_start


        if not vib_seg.empty and not optical_seg.empty:
            # 畫圖
            fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
            axes[0].plot(vib_seg["X_Time"], vib_seg["X_Amp"], label="X", color='r')
            axes[0].set_ylabel("X Amplitude (V)")
            axes[0].legend(loc="upper right")
            axes[1].plot(vib_seg["Y_Time"], vib_seg["Y_Amp"], label="Y", color='g')
            axes[1].set_ylabel("Y Amplitude (V)")
            axes[1].legend(loc="upper right")
            axes[2].plot(vib_seg["Z_Time"], vib_seg["Z_Amp"], label="Z", color='b')
            axes[2].set_ylabel("Z Amplitude (V)")
            axes[2].legend(loc="upper right")
            axes[3].plot(optical_seg["Time_shifted"], optical_seg["Optical_Power"], label="Optical Power", color='orange')
            axes[3].set_ylabel("Optical Power (dBm)")
            axes[3].set_xlabel("Time (s)")
            axes[3].legend(loc="upper right")
            plt.suptitle(f"Segment {segment}: {t_start:.2f}s to {t_end:.2f}s")
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(plot_dir, f"{SENSOR_TYPE}_{Z_CONTACT}_{segment}.png"))
            plt.close()

            # 存成 Excel
            vib_seg = vib_seg.copy()
            vib_seg["Interval_Start"] = t_start
            vib_seg["Interval_End"] = t_end
            optical_seg = optical_seg.copy()
            optical_seg["Interval_Start"] = t_start
            optical_seg["Interval_End"] = t_end

            vib_seg.to_csv(os.path.join(data_dir, f"{SENSOR_TYPE}_{Z_CONTACT}_{segment}_vibration.csv"), index=False)
            optical_seg.to_csv(os.path.join(data_dir, f"{SENSOR_TYPE}_{Z_CONTACT}_{segment}_optical.csv"), index=False)

        t += interval
        segment += 1

def main():
    parser = argparse.ArgumentParser(description="對齊震動與耦光數據並輸出圖與Excel")
    parser.add_argument("--vib_dir", type=str, required=True, help="震動數據資料夾路徑")
    parser.add_argument("--opt_file", type=str, required=True, help="耦光數據檔案路徑")
    parser.add_argument("--output_dir", type=str, required=True, help="輸出結果資料夾")
    parser.add_argument("--interval", type=int, default=2, help="對齊時間區段 (秒)，預設 2 秒")
    parser.add_argument("--sensor", type=str, required=True, help="sensor type: Probe/6Axis")
    parser.add_argument("--zcontact", type=str, required=True, help="stop/500ms/1500ms")
    args = parser.parse_args()

    global SENSOR_TYPE, Z_CONTACT
    SENSOR_TYPE = args.sensor
    Z_CONTACT = args.zcontact
    
    vib_df = load_vibration_data(args.vib_dir)
    optical_df = load_optical_data(args.opt_file)

    align_and_plot(vib_df, optical_df, args.output_dir, args.interval)

if __name__ == "__main__":
    main()
