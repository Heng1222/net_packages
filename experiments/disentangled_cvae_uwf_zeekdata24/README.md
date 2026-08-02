# UWF-ZeekData24 TTP Disentanglement

這個獨立實驗以有 ATT&CK 標註的 UWF-ZeekData24 驗證 condition 是否能拆出預期的 tactic / technique 概念。它保留 Step1 Disentangled CVAE 的 H/C reconstruction、residual adversary、condition ablation 與 latent visualization，但不修改原本的 `disentangled_cvae_step1` 實驗。

資料來源是 [UWF-ZeekData24 官方 CSV](https://datasets.uwf.edu/data/UWF-ZeekData24/csv/)；資料集方法與標註流程見[論文](https://doi.org/10.3390/data10050059)，授權為 CC BY 4.0。公開 CSV 是 Zeek connection records，不含封包 payload bytes。

## 實驗定義

- 13 個 ATT&CK tactic vectors 先扣除共同 centroid；扣除的**未正規化 centroid**直接追加為第 14 個 conditional。
- decoder、gates、ablation 都使用 14 個 conditionals；只有前 13 個 tactic logits 使用 multi-label `BCEWithLogitsLoss`，共同 conditional 不接受 tactic label supervision。
- residual adversary 同樣預測 13 維 multi-hot tactic，`pos_weight` 只用 training split 計算。
- T1078 同時合併為 Initial Access、Defense Evasion、Persistence、Privilege Escalation。這四個標籤在此資料集不可分，只能解讀為群組 alignment。
- technique probes 只使用 malicious flows，比較 `X`、前 13 個 `gates`、`C_summary`、`H` 與 `HC`。五類為 T1048、T1078、T1110、T1190、T1595。

flow text 只序列化 protocol、service、ports、connection state、history、local flags，以及 log2-bucketed duration/bytes/packet counts。UID、community ID、IP、時間與所有 label 欄位都不會進入文字 embedding。

## 執行

從 repository root 使用：

```powershell
uv run python experiments\disentangled_cvae_uwf_zeekdata24\run_experiment.py --config experiments\disentangled_cvae_uwf_zeekdata24\configs\default.yaml --stage download
uv run python experiments\disentangled_cvae_uwf_zeekdata24\run_experiment.py --config experiments\disentangled_cvae_uwf_zeekdata24\configs\default.yaml --stage prepare
uv run python experiments\disentangled_cvae_uwf_zeekdata24\run_experiment.py --config experiments\disentangled_cvae_uwf_zeekdata24\configs\default.yaml --stage train
```

也可用 `--stage all`。`--force-download` 重新下載，`--force-prepare` 忽略 prepared cache。原始 CSV 寫入 `Year=2024/UWF-ZeekData24/raw/`，下載 manifest 保存 URL、時間、大小、SHA-256、schema 與 attribution；這些資料和 `outputs/` 都不應提交 Git。

預設對 technique（含 Benign）做 deterministic 70/15/15 stratified split，同一 UID 聚合後才 split，因此不會跨 split。Benign、T1110、T1595 各最多保留 5,000 UID，其餘稀有樣本全數保留。

## 主要輸出

- `metrics/tactic_metrics.json`：validation 校準 threshold 後的一次性 test multi-label 指標與 per-tactic 結果。
- `metrics/probe_metrics.json`：frozen representation technique probes；LogisticRegression `C` 只由 validation macro-F1 選擇。
- `metrics/label_shuffle_baseline.json`、`majority_baseline.json`：基準結果。
- `metrics/technique_probe_delta_bootstrap.json`：最佳 C-derived representation 減 H 的 macro-F1 bootstrap CI。
- `metrics/reconstruction_gain_bootstrap.json`：H-only MSE 減 full MSE 的 bootstrap CI。
- `metrics/reconstruction_metrics.json`：full、H-only、C-only test MSE 與相對 full 的 bootstrap CI。
- `metrics/semantic_acceptance.json`：計畫中的全部 separation 判定條件。
- `metrics/testset_predictions.csv`：14 個 gate probabilities、13 個 gold/predicted tactics。
- `metrics/condition_ablation_summary.csv`、`condition_gate_summary.csv` 與 `plots/`：reconstruction、gate、condition geometry 與 latent diagnostics。

只有 `semantic_acceptance.json` 的全部檢查同時成立才可說 condition separation 獲得支持。若 H probe 優於 X 或最佳 C-derived representation，報告會標記 residual leakage，不宣稱成功 disentangle。這仍是受控小型概念識別實驗，不代表真實部署效能，也不涵蓋所有 13 個 tactics。

## 測試

```powershell
uv run python -m unittest discover -s experiments\disentangled_cvae_uwf_zeekdata24\tests -v
```

integration test 使用本地 synthetic CSV 與 mock downloader，不依賴網路或 ModernBERT 權重。
