# Step1 Disentangled CVAE：Payload Tactic 解耦實驗

## 1. 專案目標

這個實驗不是只訓練一個 Tactic classifier，也不是只觀察單一 latent space。
目標是將每筆 network payload embedding 分解成兩部分：

- `H`：跨 Tactic 共用或與 Tactic 無關的 residual/shared information，例如協定、
  語法結構、流量來源與 embedding 中不能由 Tactic 解釋的資訊。
- `gated conditions`：payload 對每個 MITRE ATT&CK Tactic condition 的 evidence
  scores，以及這些 condition 對 reconstruction 的實際貢獻。

期望的表示為：

```text
payload embedding x
  = shared/residual information H
  + condition-related information selected by gates
```

最終希望回答：

1. 這筆 payload 最支持哪些 Tactics？
2. 每個 gate 是否對應正確的 Tactic 語意？
3. Decoder 是否真的使用該 condition，而不是把所有資訊藏進 `H`？
4. 移除某個 condition 後，reconstruction 會惡化多少？
5. `H` 中是否仍洩漏大量 Tactic 資訊？

> 目前 gate 應稱為 **Tactic evidence score**，不是已校準的機率或百分比。
> 在 real held-out calibration 通過以前，不能解讀為「payload 含有 70% Discovery」。

本實驗獨立位於 `experiments/disentangled_cvae_step1/`，不修改既有的
`experiments/ae_cvae_tactic/` 實驗。

---

## 2. 目前進度與結論

### 2.1 已完成

- [x] Step1 CSV streaming 與 payload parsing。
- [x] ModernBERT payload embedding 與 reusable prepared cache 設計。
- [x] Payload overflow 的 `chunk_mean` 策略。
- [x] 13 個 MITRE Tactic condition texts 與 condition embeddings。
- [x] Condition common-background geometry transform。
- [x] Payload behavior projector 與 cosine sigmoid gates。
- [x] CVAE residual `H`、gated condition decoder 與 reconstruction。
- [x] Golden `Tactic` InfoNCE alignment。
- [x] H-only、C-only、condition ablation 與 gate summaries。
- [x] Residual gradient-reversal adversary，作為後續可選 regularizer。
- [x] Time split、payload-hash leakage report 與 label split coverage report。
- [x] `feasibility_baseline_v1` 最小 loss profile。
- [x] Synthetic multi-condition recovery validation。
- [x] Unit/integration tests；目前完整測試為 19 tests passing。

### 2.2 Synthetic feasibility validation

`concept_validation.py` 產生具有已知 condition mixture 與獨立 shared residual 的
synthetic data，在 train subset 訓練、在 unseen holdout 評估。

Seed 42 的已驗證結果：

| Metric | Result | Pass threshold |
| --- | ---: | ---: |
| Multi-label macro F1 | 1.0000 | >= 0.80 |
| Gate/target correlation | 0.9999 | >= 0.75 |
| Full reconstruction MSE | 0.0272 | diagnostic |
| H-only reconstruction MSE | 1.3801 | must exceed full |
| Condition reconstruction gain | 1.3529 | >= 0.05 |
| Shuffled-target macro F1 | 0.3922 | model should exceed by >= 0.20 |

這表示目前 `H + gated conditions` 實作在「可識別、可加成」的資料生成條件下，
有能力恢復 condition mixture。它不能證明真實 ModernBERT payload embedding 已經
包含可用相同方式分解的 MITRE Tactic 結構。

### 2.3 目前 real-data blocker

目前檢查到：

- Step1 prepared manifest 宣告 409,699 rows、768 dimensions。
- Prepared directory 曾只有 `metadata.csv` 與 `manifest.json`，缺少 `x.npy`；因此
  real training 尚不能視為完整執行。
- Golden review 有 2,000 rows，只有 1,196 Session IDs match prepared metadata。
- Match 後可用的 known-condition labels 約為：train 1,142、validation 35、test 0。
- `Normal (TA9000)` 不在 condition set；match rows 中有 19 筆 Normal，目前不參與
  semantic loss。
- 部分 Tactics 完全沒有 matched golden examples，部分只有 1 筆或極少數樣本。

因此目前的 time-split test set **不能驗證 Tactic classification**。UMAP、低
reconstruction loss、gate activation 或 condition geometry 都不能取代 real
held-out labels。

每次 run 會輸出 `metrics/behavior_supervision_by_split.json`。只有
`semantic_test_is_valid: true` 時，test classification metrics 才能被解讀。

---

## 3. 資料來源與角色

### 3.1 Step1 payload data

| Item | Default |
| --- | --- |
| Input | `Year=2022/Step1_rawdata_cleaned.csv` |
| Sample ID | `Session_ID` |
| Payload text | `clean_payload_list` |
| Time | `Datetime` |
| Metadata label | `Sess_Tactic_predict` |
| Split | time 70% / 15% / 15% |

`Sess_Tactic_predict` 只保留在 metadata 供 traceability 使用，不參與 condition
alignment、分類 supervision 或 condition grouping。這可避免用舊模型預測結果
自我監督後再把它當成 ground truth。

### 3.2 Golden supervision

| Item | Default |
| --- | --- |
| CSV | `Year=2022/Step2_golden_review_2_with_Tactic.csv` |
| Join key | `Session_ID` |
| Label | `Tactic` |

只有 Step2 golden review 的 `Tactic` 欄位可用於 semantic alignment。程式會拒絕
把 `Sess_Tactic_predict` 設為 supervision label。

目前 golden label 是 single-label target，因此只能直接監督「哪一個 Tactic
最符合」。若要主張 payload 同時包含多個 Tactics 以及每個 Tactic 的量，未來需要
multi-label/multi-hot review 或其他可識別的 quantitative target。

### 3.3 Normal 的定義

`Normal (TA9000)` 刻意不建立 condition vector。理想語意是所有 malicious tactic
gates 都低，但目前 Normal rows 會因 label 不在 condition set 而被 semantic loss
忽略。後續應將 reviewed Normal 明確訓練為 all-zero gate targets。

---

## 4. Prepare pipeline

### 4.1 Payload 文字轉換

Pipeline streaming 讀取大型 CSV，將 `clean_payload_list` 轉成可供 embedder 使用的
文字，並保存 sample ID、時間、payload hash、ISP、protocol 等 metadata。

### 4.2 Payload embedding

預設 embedder：

```text
nomic-ai/modernbert-embed-base
revision: d556a88e332558790b210f7bdbe87da2fa94a8d8
dimension: 768
max length: 8192
normalize: true
```

Token length 檢查結果：median 94、p99 332、p99.9 650；409,699 rows 中只有 7
rows 超過 8192 tokens，最大約 45,700。因此一般 payload 不切 chunk，只有 overflow
rows 被切分後分別 embedding，最後取 chunk mean，避免 silent truncation。

### 4.3 Prepared cache

```text
outputs/disentangled_cvae_step1/prepared/step1_clean_payload_modernbert/
├── x.npy          # [N, 768] payload embeddings
├── metadata.csv   # sample/time/hash/traceability metadata
└── manifest.json  # source/config fingerprint and expected shape
```

Manifest fingerprint 包含來源檔大小與 mtime、欄位設定、row policy 與 embedder 設定。
只有 fingerprint 相同且必要檔案完整時才可重用 cache。

### 4.4 Standardization 與 split leakage

Training stage 先做 chronological split，再只使用 train rows fit `StandardScaler`。
Val/test 不參與 scaler fitting。Standardized data 以 memmap 寫入 run directory，
避免一次把完整 409,699 x 768 matrix 複製到 RAM。

`leakage_report.json` 會用 payload hash 檢查相同 payload 是否跨越 split。這是
diagnostic，目前不會自動移除 duplicate rows。

---

## 5. Condition 建立與 geometry

### 5.1 Condition schema

Condition 定義位於：

```text
conditions/mitre_attack_v11_3_step1.yaml
```

共有 13 個 MITRE Enterprise ATT&CK Tactics，不包含 Normal。每個 condition text
由以下內容組成：

- tactic-specific keywords
- 該 tactic 下的 top-level technique names

Technique IDs、sub-techniques 與完整長描述不放進 condition text，目的是降低
MITRE 描述中大量共通詞彙造成的 embedding collapse。

### 5.2 Raw condition embedding

Condition text 使用與 payload 相同的 ModernBERT model，得到：

```text
C_raw ∈ R^(K x 768), K = 13
```

使用相同 encoder 不代表 payload text 與 MITRE prose 天然對齊，因此後面仍需要
learned payload behavior projector 與 golden InfoNCE alignment。

### 5.3 Common-component removal

預設 geometry：

```yaml
geometry:
  method: common_component_removal
  center: true
  remove_top_components: 0
  normalize: true
  strength: 1.0
```

處理流程：

1. 計算 condition centroid。
2. 從每個 condition vector 移除 shared centroid。
3. 視設定移除前幾個 principal directions。
4. Row normalization。

預設 `remove_top_components: 0`，即只做 centering + normalization。這是 fixed
post-processing，不 fine-tune ModernBERT，也不是 contrastive training。

目前 13 conditions 的 geometry diagnostic：

| Off-diagonal cosine | Raw | Model-used |
| --- | ---: | ---: |
| Mean | 0.7545 | -0.0830 |
| Median | 0.7519 | -0.0875 |
| Max | 0.8414 | 0.2850 |

轉換後略負的平均值是 centered finite set 的幾何結果，不代表 Tactics 在語意上
互相相反。Raw 與 transformed matrices 都會保存，避免只展示有利的 geometry。

---

## 6. 完整模型架構

### 6.1 End-to-end data flow

```text
Payload text
  │
  ├─ ModernBERT ──> raw payload embedding x_raw [768]
  │
  └─ train-fitted StandardScaler ──> x [768]
                                      │
                  ┌───────────────────┴───────────────────┐
                  │                                       │
                  ▼                                       ▼
        Residual CVAE encoder                    Payload behavior projector
        concat(x, flatten(C))                    MLP(x) -> normalize
                  │                                       │
             μ_H, logσ²_H                                 q [768]
                  │                                       │
          reparameterization                              │ cosine(q, C_i)
                  │                                       ▼
                H [64]                            independent sigmoid gates
                  │                                       │
                  └──────────────┬────────────────────────┘
                                 ▼
                      flatten(g_i * C_i) for all i
                                 │
                     concat(H, gated conditions)
                                 │
                            MLP decoder
                                 │
                                 ▼
                         reconstructed x_hat [768]
```

### 6.2 Payload behavior projector

```text
q = normalize(Projector(x))
```

用途：

- 將 standardized payload embedding 映射到 transformed condition geometry。
- 不假設 raw payload embedding 與 MITRE condition embedding 有相同 centroid。
- 讓 golden supervision 可以直接調整 payload-to-condition alignment。

目標：正確 condition 的 cosine score 應高於其他 conditions。

### 6.3 Gate computation

對每個 condition：

```text
s_i = cosine(q, C_i)
g_i = sigmoid(s_i / temperature)
```

選擇 independent sigmoid 而不是 softmax，因為一筆 payload 理論上可同時包含多個
Tactics。`temperature` 控制 sigmoid sharpness。

目前限制：

- 沒有 per-condition calibrated bias。
- `0.5` threshold 對應 cosine `0`，不是由 held-out data 校準而來。
- Single-label InfoNCE 只約束 target 相對排名，不等於完整 multi-label calibration。

### 6.4 Residual CVAE encoder

目前 encoder input：

```text
encoder_input = concat(x, flatten(C_all))
```

輸出 `μ_H`、`logσ²_H`，並使用 reparameterization 得到：

```text
H = μ_H + ε * exp(0.5 * logσ²_H)
```

`H` 的用途是保存無法由 named conditions 解釋、但 reconstruction 仍需要的 shared
或 residual information。

目前已知限制：`C_all` 對所有 samples 都相同，因此 concatenating flattened C
主要等價於加入大量固定 bias，並沒有提供 sample-specific condition information。
後續應比較只輸入 `x`，或輸入 `[x, gates]` 的較小 encoder。

### 6.5 Gated condition representation

```text
C_gated_i = g_i * C_i
```

Evaluation 用的 semantic summary 是 gated conditions 的 normalized weighted mean：

```text
C_summary = sum_i(g_i C_i) / max(sum_i(g_i), epsilon)
```

用途：建立可視化與 downstream representation，觀察 condition pathway 聚合後的
語意空間。

### 6.6 Decoder

```text
decoder_input = concat(H, flatten(gates * C_all))
x_hat = Decoder(decoder_input)
```

用途：要求 reconstruction 同時依賴 residual `H` 與 selected conditions。

目前已知限制：因為 `C_all` 是 fixed matrix，第一層 linear decoder 對
`flatten(g_i C_i)` 的操作，在代數上近似對 K 個 gate scalars 的大型 learned
transform。未來可直接比較 `[H, gates]` 或先把每個 `C_i` 壓到小型 concept vector，
以降低 13 x 768 condition input 的參數量並讓 bottleneck 更明確。

### 6.7 Residual adversary

```text
H -> Gradient Reversal -> linear Tactic classifier
```

Classifier 本身學習從 `H` 預測 Tactic；gradient reversal 讓 encoder 接收反向梯度，
嘗試移除 H 中的 Tactic 資訊。

用途：降低 residual leakage。

限制：目前 adversary 只有 linear layer，不能證明 nonlinear tactic information 已被
移除。正式結果仍需 frozen post-hoc nonlinear probe。

---

## 7. Loss functions：用途、目標與風險

總 loss：

```text
L_total =
    w_rec       * L_rec
  + w_kl        * L_kl
  + w_decor     * L_decorrelation
  + w_sparse    * L_sparse
  + w_entropy   * L_gate_entropy
  + w_utility   * L_utility
  + w_residual  * L_residual_constraint
  + w_behavior  * L_behavior_infonce
  + w_adversary * L_residual_adversary
```

### 7.1 Reconstruction NLL

```text
L_rec = mean_n 0.5 * sum_d[
  (x_nd - x_hat_nd)^2 / observation_variance
  + log(2π * observation_variance)
]
```

用途：保留 payload embedding information，避免 gate 只為了分類而與原始 payload
無關。

注意：它對 768 dimensions 求和，loss scale 會比一般 mean MSE 大；調整其他 loss
weights 時必須看實際數量級。

### 7.2 KL divergence on H

```text
L_kl = -0.5 * sum_j(1 + logσ²_j - μ_j² - exp(logσ²_j))
```

用途：限制 H capacity，讓 H 接近 standard normal prior，降低 encoder 把全部 payload
記憶在 residual channel 的傾向。

風險：過強會 posterior collapse；過弱則 H 可能繞過 condition pathway。

### 7.3 Gate decorrelation/co-activation penalty

實作以 condition cosine squared 乘上 gate pair activation，懲罰相似 conditions 同時
啟動。

用途：減少 highly similar conditions 重複解釋相同 payload。

限制：condition vectors 是 fixed，這個 loss 不會真正改變 condition geometry；它只
抑制 co-activation。Cosine squared 也會懲罰負相關 conditions，且合理的 multi-tactic
payload 本來就可能需要多個 gates。因此 baseline 與建議 `full_v1` 暫時設為 `0`。

### 7.4 Gate sparsity

```text
L_sparse = mean(gates)
```

用途：實作「沒有必要就不要拆出 condition」，鼓勵較少 active Tactics。

風險：過強會造成 all-zero collapse，尤其目前 golden labels 很稀疏。

### 7.5 Gate entropy

```text
L_entropy = mean[-g log(g) - (1-g) log(1-g)]
```

Minimize 時會鼓勵 gates 接近 0 或 1。

用途：避免大量 scores 長期停在 0.5 附近。

風險：它只增加 confidence，不保證 confident prediction 是正確的。

### 7.6 Per-condition utility margin

對每個 condition i：

```text
delta_i = MSE(x_hat_without_i, x) - MSE(x_hat_full, x)
L_utility = gate-weighted ReLU(utility_margin - delta_i)
```

用途：若 gate 啟動，移除該 condition 應使 reconstruction 變差，否則該 gate 可能只是
裝飾。

風險：要求每個 active condition 都達到固定 margin 可能造成冗餘，且每 batch 需要 K
次額外 decoder forward。當 utility weight 為 0 時，training 會跳過這些 ablations；
final evaluation 仍會計算真實 deltas。

### 7.7 Residual/condition-use constraint

```text
L_residual = ReLU[
  residual_margin - (MSE_H_only - MSE_full)
]
```

用途：要求拿掉所有 condition gates 後，H-only reconstruction 至少惡化一個 margin，
防止 decoder 完全忽略 condition pathway。

風險：zero gates 是 out-of-distribution intervention；未來可加入 condition dropout，
讓模型在 training 中實際看過此情況。

### 7.8 Golden behavior InfoNCE

只對成功 join 且 label 屬於 condition set 的 rows：

```text
logits_i = cosine(q, C_i) / behavior_temperature
L_behavior = CrossEntropy(logits, golden_tactic_index)
```

用途：將每個 gate dimension anchor 到具名 MITRE Tactic，避免 reconstruction-only
decomposition 的 permutation/non-identifiability。

限制：目前是 single-label CE，只監督 top-1 prototype ranking；不能驗證多 Tactic
amounts，也沒有直接使用 reviewed Normal 作 all-zero targets。

### 7.9 Residual adversarial loss

```text
L_adversary = CrossEntropy(Adversary(GRL(H)), golden_tactic)
```

用途：adversary 學會預測 Tactic，同時 GRL 讓 encoder 反向降低 H 的 tactic
predictability。

風險：過早啟用會與 reconstruction、semantic alignment 競爭，因此 baseline 關閉，
待基本對齊成立後才作 ablation。

---

## 8. 分階段訓練策略

### 8.1 Phase A：Feasibility baseline

Default config 使用：

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

這個 baseline 只回答四個最小問題：

1. `x_hat` 能否重建 payload embedding？
2. Payload behavior query 能否對準 reviewed Tactic？
3. Full reconstruction 是否優於 H-only？
4. Gates 是否具有 sample-specific variation，而不是 collapse？

若 baseline 未通過，不應加入更多 regularizers，因為那會讓失敗原因更難定位。

### 8.2 Phase B：逐項 ablation

建議順序：

1. Baseline + residual adversary `0.1`。
2. Baseline + sparse `0.01`。
3. Baseline + gate entropy `0.01`。
4. Baseline + utility `0.1`。
5. 比較 condition dropout、較小 H 與較小 decoder condition representation。

每次只增加一項，固定 data split、seed、embedding、architecture 與 evaluation。

### 8.3 Phase C：Full v1 候選

Baseline 與單項 ablations 通過後，可測試：

```yaml
loss_profile: full_v1
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

這是第一個組合候選，不是已證明最佳參數。`decorrelation` 維持 0，直到有證據顯示
它能改善 real held-out semantic/faithfulness metrics，而不是只讓 gates 更少。

---

## 9. Training 與 checkpoint 邏輯

- Optimizer：AdamW。
- Train loader：shuffle，固定 seed。
- Validation：不 shuffle，使用 `sample=False`，即 `H = μ_H`。
- Early stopping：監控 total validation loss。
- Best checkpoint：只保留最低 validation loss 的 model/optimizer state。
- Training 結束後重新載入 best checkpoint，再執行 test extraction。
- Per-epoch history 保存 train/val 各 loss、reconstruction、H-only/C-only MSE 與
  labeled accuracy/count。

目前 validation golden rows 很少，因此 total val loss 主要由 unsupervised
reconstruction/KL 決定；即使 best val loss 很低，也不代表 semantic alignment 最佳。
未來應加入具備足夠 labels 的 semantic validation 或另外的 model-selection metric。

---

## 10. Evaluation 設計與成功判準

### 10.1 Reconstruction fidelity

- `recon_mse`：full H + gated conditions。
- `h_only_mse`：所有 gates 設為 0。
- `c_only_mse`：H 設為 0。

必要但不充分的條件：

```text
h_only_mse > recon_mse
h_only_mse - recon_mse > 0
```

這只能證明 condition pathway 有 reconstruction utility，不能證明 Tactic 名稱正確。

### 10.2 Condition ablation

逐一把 `g_i` 設為 0，計算：

```text
delta_mse_i = ablated_mse_i - full_mse
```

Active/important condition 預期有正 delta。負值代表移除 condition 反而改善結果，
可能是 gate 錯誤、decoder 干擾或 condition 沒有被正確使用。

### 10.3 Semantic alignment

在 test predictions 與 golden labels 上輸出：

- accuracy
- macro F1
- weighted F1
- per-class precision/recall/F1
- non-ambiguous rate

只有 test 有 golden support 時可解讀。正式比較至少需要：

- majority-class baseline
- 相同 payload embeddings 的 plain classifier baseline
- shuffled-label baseline
- 建議補 macro-AUPRC 與 calibration metrics

### 10.4 Gate behavior

每個 Tactic 輸出：

- mean/std gate
- p50/p90/p99
- threshold active rate
- 每筆 active condition count

需排除：all-zero、all-one、constant-prior 與 rare-class never-active collapse。

### 10.5 Residual leakage

Training 期間會輸出 adversary accuracy，但正式判斷應在 frozen H 上另外訓練
post-hoc nonlinear probe。理想狀況是 gate space 保有 tactic predictability，H probe
接近 declared baseline。

### 10.6 Condition geometry

保存 raw/transformed cosine matrices 與 heatmaps。Geometry spread 是必要 diagnostic，
但不是 payload alignment 的證據。

### 10.7 UMAP/PCA

輸出 original x、H、gated C space 的 2D projection。它只能作探索性輔助，不能作
主要成功判準，尤其目前 plot 顏色可能來自模型自己的 predicted condition。

---

## 11. 執行命令

安裝環境與測試：

```powershell
uv sync
uv run python -m unittest discover `
  -s experiments\disentangled_cvae_step1\tests -v
```

Synthetic prerequisite validation：

```powershell
uv run python -m experiments.disentangled_cvae_step1.concept_validation `
  --seed 42 `
  --epochs 80 `
  --output outputs\disentangled_cvae_step1\concept_validation.json
```

Prepare embeddings：

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage prepare
```

Prepared cache 明確損壞時才強制重建：

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage prepare `
  --force-prepare
```

重用 embeddings，執行 baseline training：

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage train
```

Prepare + baseline training：

```powershell
uv run python experiments\disentangled_cvae_step1\run_experiment.py `
  --config experiments\disentangled_cvae_step1\configs\default.yaml `
  --stage all
```

完整逐步測試、log 觀察、`full_v1` 建立方式與 Go/No-Go checklist 請見 repository
root 的 [`todo.md`](../../todo.md)。

---

## 12. Outputs

每次 run 建立：

```text
outputs/disentangled_cvae_step1/<timestamp>/
├── config_resolved.yaml
├── environment.json
├── logs/
│   └── experiment.log
├── checkpoints/
│   └── disentangled_cvae.pt
├── scalers/
│   ├── scaler.pkl
│   └── x_standard.npy
├── embeddings/
│   ├── condition_embeddings.npz
│   └── condition_embeddings_metadata.json
├── metrics/
│   ├── split_assignments.csv
│   ├── leakage_report.json
│   ├── behavior_supervision_summary.json
│   ├── behavior_supervision_by_split.json
│   ├── training_history.csv
│   ├── training_summary.json
│   ├── loss_summary.json
│   ├── behavior_alignment_metrics.json
│   ├── condition_gate_summary.csv
│   ├── condition_ablation_delta_mse_summary.csv
│   ├── condition_raw_cosine_similarity.csv
│   ├── condition_cosine_similarity.csv
│   ├── testset_condition_predictions.csv
│   └── testset_subset_100.csv
├── plots/
│   ├── condition_raw_cosine_similarity.png
│   ├── condition_cosine_similarity.png
│   ├── training_reconstruction_losses.png
│   ├── umap_original_space.png
│   ├── umap_h_space.png
│   └── umap_gated_c_space.png
└── reports/
    └── report.md
```

`testset_condition_predictions.csv` 包含 metadata、gold Tactic、每個 condition score、
最高 condition、threshold 後的 multi-condition 字串與 active condition count。

---

## 13. 程式結構

```text
disentangled_cvae_step1/
├── configs/default.yaml          # feasibility baseline config
├── conditions/                   # MITRE condition definitions
├── data.py                       # prepare/cache/split/supervision/scaling
├── embedders.py                  # ModernBERT and hashing embedders
├── conditions.py                 # condition loading/geometry/cache
├── model.py                      # CVAE, gates, losses, adversary
├── training.py                   # train/early stop/extraction
├── evaluate.py                   # metrics, CSVs, plots
├── concept_validation.py         # synthetic feasibility check
├── run_experiment.py             # prepare/train/all CLI
├── PROGRESS_REVIEW.md            # architecture audit and recommendations
└── tests/                         # unit and integration tests
```

---

## 14. 已知限制與下一步

優先順序：

1. **補齊 prepared `x.npy`**，驗證 x/metadata/manifest row alignment。
2. **建立有效 real test labels**，尤其補足 chronological test window 與 unsupported
   Tactics。
3. **建立 plain classifier baseline**，使用完全相同的 fixed payload embeddings 與
   split，避免只和 random/shuffle 比較。
4. **先跑 feasibility baseline**，確認 semantic alignment 與 condition-use gain。
5. **Normal all-zero supervision**，並支援真正的 multi-label targets/BCE。
6. **簡化 encoder/decoder condition input**，比較 `[x]` encoder 與 `[H, gates]`
   decoder，移除 fixed flattened condition matrix 的冗餘。
7. **加入 calibrated gate bias/threshold**，在 held-out labels 上評估 Brier score、
   ECE、AUROC/AUPRC。
8. **Frozen nonlinear H leakage probe**，而不是只看 training adversary。
9. **逐項 loss ablation**，最後才組合 full model。
10. **至少三個 seeds**，回報 mean ± std，不使用單次最佳 run 作結論。

只有同時滿足以下條件，才能主張真實 payload Tactic 解耦有初步成效：

- Real held-out semantic metrics 明顯優於 majority/plain-classifier/shuffle baselines。
- Full reconstruction 明顯優於 H-only。
- Active condition ablation 有正且穩定的 delta。
- Gates 不 collapse，並具 sample-specific variation。
- H 的 post-hoc tactic leakage 顯著低於 original embedding/gate space。
- 結果能跨 seeds 重現。

更完整的模型審查與設計風險請見 [`PROGRESS_REVIEW.md`](PROGRESS_REVIEW.md)。
