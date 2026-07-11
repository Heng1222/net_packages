# Disentangled CVAE Step1：測試與訓練操作清單

這份文件的目標是依序回答以下問題：

1. 專案、資料與 embedding cache 是否完整？
2. `H + gated conditions` 架構在可識別資料上是否做得到解耦？
3. 最小 loss baseline 在真實 payload 上，gate 是否對準 Tactic？
4. Decoder 是否真的使用 condition，而不是把所有資訊藏進 `H`？
5. Baseline 通過後，加入 regularization 的完整模型是否真的更好？

> 重要：gate 目前只能稱為 **Tactic evidence score**。在通過 held-out
> calibration 以前，不可解讀為「payload 有 70% Discovery」。

---

## 0. 固定實驗條件

所有命令都在 repository root 執行：

```powershell
Set-Location "C:\Users\user\Desktop\Lab\net packages"
uv sync
```

確認 PyTorch 是否看到 GPU：

```powershell
uv run python -c "import torch; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

執行完整測試：

```powershell
uv run python -m unittest discover `
  -s experiments\disentangled_cvae_step1\tests -v
```

通過條件：

- 所有 tests 都是 `OK`。
- `test_recovers_known_condition_mixtures` 必須通過。
- `test_smoke_prepare_train_report` 必須通過。

若 tests 未通過，不要開始正式 embedding 或 training。

---

## 1. 檢查輸入資料與 prepared cache

### 1.1 檢查必要檔案

```powershell
Get-Item `
  Year=2022\Step1_rawdata_cleaned.csv, `
  Year=2022\Step2_golden_review_2_with_Tactic.csv, `
  experiments\disentangled_cvae_step1\configs\default.yaml
```

Prepared dataset 必須同時存在以下三個檔案：

```powershell
$prepared = "outputs\disentangled_cvae_step1\prepared\step1_clean_payload_modernbert"
Get-Item `
  "$prepared\x.npy", `
  "$prepared\metadata.csv", `
  "$prepared\manifest.json"
```

目前已知狀態：`metadata.csv` 與 `manifest.json` 存在，但 `x.npy` 曾經缺失。
只要 `x.npy` 不存在，就不能執行 `--stage train`。

### 1.2 建立或補齊 payload embeddings

第一次執行可能下載 ModernBERT weights，且完整 409,699 rows 需要較長時間：

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage prepare
```

若 cache 已損壞或明確需要重建：

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage prepare `
  --force-prepare
```

不要在 embedding 正常進行時刪除 prepared directory。

### 1.3 驗證 cache row alignment

```powershell
uv run python -c "import json,numpy as np,pandas as pd; p=r'outputs/disentangled_cvae_step1/prepared/step1_clean_payload_modernbert'; x=np.load(p+'/x.npy',mmap_mode='r'); m=pd.read_csv(p+'/metadata.csv',usecols=['sample_id']); manifest=json.load(open(p+'/manifest.json',encoding='utf-8')); print('x=',x.shape,'metadata=',len(m),'manifest_rows=',manifest['rows']); assert len(x)==len(m)==manifest['rows']; assert x.shape[1]==768"
```

通過條件：

- `x.shape == (409699, 768)`，或與最新 manifest 宣告完全一致。
- `len(metadata) == manifest["rows"] == len(x)`。
- 任一數字不一致時，使用 `--force-prepare` 重建，不要繼續 training。

---

## 2. 先跑 synthetic concept validation

這一步不是驗證真實 MITRE 分類，而是確認目前程式有能力從「已知 condition
mixture + shared residual」中恢復 gates。

```powershell
uv run python -m experiments.disentangled_cvae_step1.concept_validation `
  --seed 42 `
  --epochs 80 `
  --output outputs\disentangled_cvae_step1\concept_validation.json
```

檢查結果：

```powershell
Get-Content outputs\disentangled_cvae_step1\concept_validation.json
```

固定通過門檻：

- `passed: true`
- `macro_f1 >= 0.80`
- `gate_target_correlation >= 0.75`
- `condition_reconstruction_gain >= 0.05`
- `macro_f1` 至少比 `shuffled_target_macro_f1` 高 `0.20`

若失敗，代表 gate/H 實作或訓練路徑本身有問題，不應開始真實資料調參。

---

## 3. Golden label 是正式訓練前的硬性檢查

目前曾觀察到的可用 label 分布：

| Split | 可用 golden condition labels |
| --- | ---: |
| train | 1,142 |
| validation | 35 |
| test | 0 |

因此，目前的 time-split test set **不能驗證 Tactic classification**。即使 UMAP
很好看、reconstruction 很低或 gates 有變化，也不能宣稱解耦成功。

正式判讀 real-data baseline 前，至少要完成其中一種方案：

- 推薦：補標 chronological test 時段的 payload，讓 test 中每個目標 Tactic
  有足夠樣本。
- 暫時方案：另外建立 stratified golden holdout，而且 holdout labels 絕不可
  參與 training；報告必須註明它不是 time-generalization test。
- 樣本極少或完全沒有 label 的 Tactic，第一輪先排除或列為 unsupported，不能
  把 zero-support class 一起平均後宣稱 macro 指標有效。

每次完成 training 後檢查：

```powershell
$run = Get-ChildItem outputs\disentangled_cvae_step1 -Directory |
  Where-Object Name -ne "prepared" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Get-Content "$($run.FullName)\metrics\behavior_supervision_by_split.json"
```

正式 semantic test 的硬性門檻：

- `semantic_test_is_valid` 必須是 `true`。
- `test.usable_condition_labels` 必須大於 `0`。
- 需要檢查 `counts_by_label`，不能只看總數。

若為 `false`，該次 run 只能檢查工程流程與 reconstruction，不能判斷分類效果。

---

## 4. 啟動最小 Baseline

目前 [default.yaml](experiments/disentangled_cvae_step1/configs/default.yaml) 使用：

```yaml
loss_profile: feasibility_baseline_v1
weights:
  reconstruction: 1.0
  kl: 1.0
  decorrelation: 0.0
  sparse: 0.0
  gate_entropy: 0.0
  utility: 0.0
  residual_constraint: 0.1
  behavior_infonce: 1.0
  residual_adversary: 0.0
```

Baseline 只回答：

1. Payload behavior projector 能否對準 golden Tactic？
2. Condition pathway 是否改善 reconstruction？
3. `H` 是否成為合理 residual，而不是包含全部資訊？

### 4.1 只執行 training（推薦）

確認 `x.npy` 完整後：

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage train
```

### 4.2 從 prepare 到 train 一次執行

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage all
```

若 prepared fingerprint 相同，`--stage all` 應重用 cache；不要加
`--force-prepare`，除非確定要重新 embedding。

### 4.3 即時觀察 training

另開 PowerShell：

```powershell
$run = Get-ChildItem outputs\disentangled_cvae_step1 -Directory |
  Where-Object Name -ne "prepared" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

$run.FullName
Get-Content "$($run.FullName)\logs\experiment.log" -Wait
```

每個 epoch 主要觀察：

- `train_loss`、`val_loss`：應為 finite，不能出現 `nan`/`inf`。
- `train_recon_mse`、`val_recon_mse`：應下降後趨於穩定。
- `val_h_only_mse`：理想上高於 `val_recon_mse`。
- `val_c_only_mse`：通常可高於 full MSE，但不能完全沒有資訊。
- `val_behavior_acc`：只能在 validation 有 golden labels 時解讀。
- `val_behavior_labeled_count`：太少時 accuracy 波動不可信。
- Early stopping 應保留最低 `val_loss` checkpoint。

---

## 5. Baseline 結果檢查

先取得最新 run：

```powershell
$run = Get-ChildItem outputs\disentangled_cvae_step1 -Directory |
  Where-Object Name -ne "prepared" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1
```

### 5.1 確認 run 完整

```powershell
Get-Item `
  "$($run.FullName)\checkpoints\disentangled_cvae.pt", `
  "$($run.FullName)\metrics\training_history.csv", `
  "$($run.FullName)\metrics\training_summary.json", `
  "$($run.FullName)\metrics\loss_summary.json", `
  "$($run.FullName)\metrics\behavior_alignment_metrics.json", `
  "$($run.FullName)\reports\report.md"
```

任一關鍵檔案缺失，都代表 run 未完成。

### 5.2 檢查 reconstruction 與 condition contribution

```powershell
Get-Content "$($run.FullName)\metrics\loss_summary.json"
Get-Content "$($run.FullName)\metrics\condition_ablation_delta_mse_summary.csv"
```

必要關係：

```text
h_only_mse > recon_mse
condition_reconstruction_gain = h_only_mse - recon_mse > 0
```

解讀方式：

- `h_only_mse` 幾乎等於 `recon_mse`：decoder 幾乎只依賴 `H`，condition
  pathway 沒有實質貢獻。
- 多數 condition 的 `mean_delta_mse <= 0`：移除 condition 不會傷害結果，gate
  可能只是裝飾。
- `c_only_mse` 很低而 full 沒改善：可能 `H` 沒有學到 shared residual。
- Reconstruction 改善只能證明 condition 有 utility，不能證明 Tactic 名稱正確。

### 5.3 檢查 Tactic alignment

```powershell
Get-Content "$($run.FullName)\metrics\behavior_supervision_by_split.json"
Get-Content "$($run.FullName)\metrics\behavior_alignment_metrics.json"
```

只有 `semantic_test_is_valid: true` 才可解讀：

- `accuracy`
- `macro_f1`
- `weighted_f1`
- 每一類 precision / recall / F1
- `non_ambiguous_rate`

不能只看 accuracy。至少要與以下 baseline 比較：

- 預測最多數 Tactic 的 majority baseline。
- 使用相同 payload embeddings 的 plain classifier baseline。
- shuffled golden labels baseline。

預期條件：模型 macro-F1/AUPRC 應明顯優於上述 baseline，而且不是只靠
Discovery/Credential Access 等多數類別。

### 5.4 檢查 gates

```powershell
Import-Csv "$($run.FullName)\metrics\condition_gate_summary.csv" |
  Format-Table condition,mean_gate,std_gate,p50_gate,p90_gate,active_rate

Import-Csv "$($run.FullName)\metrics\testset_condition_predictions.csv" |
  Select-Object -First 20 sample_id,gold_tactic,predicted_condition,active_condition_count
```

異常模式：

- 所有 `mean_gate` 接近 `0`：gate collapse / sparsity 過強 / alignment 無效。
- 所有 `mean_gate` 接近 `1`：gate 沒有選擇性。
- `std_gate` 幾乎是 `0`：gate 只學到 condition prior，不是 sample-specific。
- 每筆 `active_condition_count` 都相同：需要檢查 threshold 或 gate collapse。
- Rare tactics 永遠不啟動：先檢查 golden support，不要直接增加 loss weight。

### 5.5 檢查 condition geometry

```powershell
Get-Content "$($run.FullName)\metrics\condition_raw_cosine_similarity.csv" -TotalCount 5
Get-Content "$($run.FullName)\metrics\condition_cosine_similarity.csv" -TotalCount 5
```

並查看：

- `plots/condition_raw_cosine_similarity.png`
- `plots/condition_cosine_similarity.png`

轉換後 conditions 不應全部高度相似；但低 cosine 只表示 geometry 被展開，
不等於 payload 已成功分類。

### 5.6 UMAP/PCA 只能作輔助

查看：

- `plots/umap_original_space.png`
- `plots/umap_h_space.png`
- `plots/umap_gated_c_space.png`

UMAP 不可作為主要成功判準。顏色分群可能來自模型自己的預測，不能取代
golden-label metrics、ablation 或 leakage probe。

---

## 6. Baseline 的 Go / No-Go 判定

只有同時符合以下條件，才進入完整模型：

- [ ] Synthetic concept validation 通過。
- [ ] Prepared `x.npy`、metadata、manifest rows 完全一致。
- [ ] Real semantic test 有足夠且跨 Tactic 的 golden labels。
- [ ] Baseline macro-F1/AUPRC 優於 majority、plain classifier、shuffle baseline。
- [ ] `h_only_mse - recon_mse > 0`，且差異不是數值雜訊。
- [ ] Active condition ablation 大多造成正的 reconstruction delta。
- [ ] Gates 有 sample-specific variation，沒有全 0、全 1 或 constant collapse。
- [ ] 多個 seeds 得到方向一致的結果。

建議至少跑三個 seeds：`42`、`43`、`44`。每個 seed 使用獨立 config，保持
資料 split、模型與 loss 不變，只修改 `seed` 與 `evaluation.random_state`。

若 semantic test 仍為 0 labels，判定必須是 **No-Go：缺少驗證資料**，不是模型
成功或失敗。

---

## 7. 建立完整模型設定 `full_v1`

不要直接覆寫 baseline config，先複製：

```powershell
Copy-Item `
  experiments\disentangled_cvae_step1\configs\default.yaml `
  experiments\disentangled_cvae_step1\configs\full_v1.yaml
```

在 `full_v1.yaml` 修改：

```yaml
model:
  loss_profile: "full_v1"
  utility_margin: 0.1
  residual_margin: 0.1
  weights:
    reconstruction: 1.0
    kl: 1.0
    decorrelation: 0.0
    sparse: 0.01
    gate_entropy: 0.01
    utility: 0.1
    residual_constraint: 0.1
    behavior_infonce: 1.0
    residual_adversary: 0.1
```

說明：

- `decorrelation` 暫時維持 `0`。目前實作會懲罰 condition co-activation，可能
  錯殺合理的 multi-tactic payload。
- `sparse` 從 `0.01` 開始，不要直接回到先前的 `1.0`。
- `utility` 與 adversary 先使用 `0.1`，避免一次壓過主 loss。
- 每新增一個 loss，理想上都應先做單獨 ablation；`full_v1` 是 baseline 通過後
  的第一個組合實驗，不是最終最佳參數。

---

## 8. 完整模型訓練指令

只重用 prepared embeddings 進行 training：

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\full_v1.yaml `
  --stage train
```

從資料準備到訓練完整執行：

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\full_v1.yaml `
  --stage all
```

訓練過程使用與 baseline 相同的 log 觀察方式。不要同時改 split、embedding、
hidden dimensions、loss weights 與 seed，否則無法知道改善來自哪裡。

---

## 9. Baseline 與 Full 結果比較

比較時固定：

- 同一份 `x.npy`
- 同一 condition embeddings 與 geometry
- 同一 train/val/test split
- 同一 seed
- 同一 golden labels
- 同一 evaluation threshold

建立比較表：

| Metric | Baseline | Full v1 | 判讀方向 |
| --- | ---: | ---: | --- |
| test macro-F1 / macro-AUPRC |  |  | 越高越好 |
| weighted-F1 |  |  | 輔助，不能取代 macro |
| non-ambiguous rate |  |  | 避免全部 ambiguous |
| full reconstruction MSE |  |  | 越低越好 |
| H-only MSE |  |  | 應高於 full |
| H-only minus full MSE |  |  | 正值且合理增大 |
| C-only MSE |  |  | 檢查 condition utility |
| mean active condition count |  |  | 不應 collapse |
| gate std / active rate |  |  | 應具 sample variation |
| post-hoc H leakage probe |  |  | 越接近 declared baseline 越好 |
| seed mean ± std |  |  | 越穩定越好 |

Full v1 只有在以下情況才算改善：

- Semantic metric 沒有下降，最好有穩定提升。
- H leakage 降低。
- Condition ablation/condition-use gain 提升。
- Gates 更稀疏或清楚，但沒有犧牲 rare-class recall。
- 改善在多個 seeds 都出現，而不是單次最佳結果。

若 reconstruction 變好但 semantic metric 下降，不算 disentanglement 改善。
若 gates 更 sparse 但全數變成 0，也不算改善。

---

## 10. 每次正式實驗要保存的資料

- [ ] 使用的 YAML config。
- [ ] Git commit 或 `git diff` 狀態。
- [ ] `environment.json`。
- [ ] `training_history.csv` 與 checkpoint。
- [ ] `behavior_supervision_by_split.json`。
- [ ] `behavior_alignment_metrics.json`。
- [ ] `loss_summary.json`。
- [ ] Gate summary 與 test predictions。
- [ ] Condition ablation summary。
- [ ] Condition geometry matrices/plots。
- [ ] 三個以上 seeds 的 mean ± std。
- [ ] Baseline 與 full_v1 的比較表。
- [ ] 清楚記錄 unsupported/zero-support Tactics。

最終報告必須分開陳述：

1. Classification/alignment 是否正確。
2. Condition 是否真的被 decoder 使用。
3. `H` 是否仍洩漏 Tactic。
4. Gate 是否穩定、可校準、可跨 seed 重現。
5. 哪些結論只是 synthetic capability，哪些來自 held-out real payload。
