# Vibration_testing
## Step
1. 將震動數據與耦光數據對齊，以每兩秒為一個單位畫出一張時域圖
2. 利用交叉相關分析(Cross-Correlation)找出震動和耦光之間的關係


### 1. 將震動數據與耦光數據對齊，以每兩秒為一個單位畫出一張時域圖
Step0.py
input: 
```
python Step0.py --vib_dir "震動檔案路徑"\
                   --opt_file "耦光檔案路徑.csv" \
                   --output_dir "輸出檔案路徑"\
                   --interval 2\
                   --sensor Probe/6Axis\
                   --zcontact stop/500ms/1500ms\
```