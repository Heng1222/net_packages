# Golden-only Oracle / Predicted-Gate CVAE 前導實驗

## 1. 實驗目的

這個實驗只使用已人工標記的：

```text
Year=2022/Step2_golden_review_2_with_Tactic.csv
```

它是完整 Step1 409,699 rows 訓練之前的 feasibility experiment，目的是用小型、
有正確答案的資料，先把三個問題拆開驗證：

1. **Condition pathway capacity**：已知正確 Tactic 時，把對應 condition 餵給
   decoder，是否真的能改善 payload embedding reconstruction？
2. **Payload-to-condition prediction**：不把正確答案餵給模型時，模型是否能從
   payload embedding 預測正確的 Tactic gates？
3. **CVAE 是否值得保留**：Predicted-gate CVAE 是否至少能接近相同 embeddings
   上的 plain classifier，同時讓 condition 對 reconstruction 有可測量的貢獻？

這個實驗不追求最終 production accuracy，也不能取代 chronological Step1 test。
它的用途是先決定整條 `payload -> gate -> H + condition -> reconstruction` 流程
是否值得投入大量標記與完整資料訓練。

此實驗位於獨立 package：

```text
experiments/golden_oracle_cvae_step2/
```

它有自己的 config、prepared cache、models、training、metrics、plots、reports 和
tests，輸出寫到 `outputs/golden_oracle_cvae_step2/`。它不修改既有
`disentangled_cvae_step1` 或 `ae_cvae_tactic` 實驗；只重用既有且不改動的 text
embedder、condition loader、plot 與通用 output helpers。

---

## 2. 核心實驗設計

同一份 golden-only train/validation/test split 會訓練三個模型，並在 test 做四種
condition controls。

### 2.1 Plain payload classifier

```text
payload embedding x
  -> MLP classifier
  -> Tactic / Normal
```

用途：建立最基本的 supervised classification baseline。如果 predicted-gate CVAE
遠低於 plain classifier，問題通常在 gate geometry、CVAE objective 或 condition
representation，而不是 payload embedding 完全沒有分類資訊。

Test inference 不會收到 gold Tactic。

### 2.2 Oracle Conditional VAE

```text
payload embedding x -> Encoder -> residual H
gold Tactic          -> one-hot gate -> gold condition summary

concat(H, gold condition summary) -> Decoder -> reconstructed x
```

用途：測量「如果 condition 選擇完全正確」，conditional decoder 最多能得到多少
幫助。這是 teacher forcing / oracle upper bound。

Oracle 成功只能證明 condition pathway 有 capacity，不能證明模型能自行分類，因為
test 時 gold Tactic 被直接餵給 decoder。

### 2.3 Predicted-Gate CVAE

```text
payload embedding x
  ├─ Encoder -> residual H
  └─ Behavior projector -> cosine with conditions -> sigmoid gates

concat(H, predicted condition summary) -> Decoder -> reconstructed x
```

Gold Tactic 只用於 train/validation 的 gate supervision。Test inference 完全不把
答案餵給模型，因此這才是接近完整 Step1 inference 的前導模型。

### 2.4 Test controls

Oracle model 會比較：

```text
A. Gold condition
B. Zero condition
C. Shuffled condition
```

Predicted model會比較：

```text
D. Predicted condition
E. Zero condition
```

這些 controls 分別回答：

- Gold 是否優於完全沒有 condition？
- Gold 是否優於來自其他 test sample 的錯誤 condition？
- Predicted condition 是否真的被 decoder 使用？
- Oracle 與 predicted condition 之間還有多少差距？

---

## 3. 資料處理

### 3.1 使用欄位

Default config 使用：

| Role | Column |
| --- | --- |
| Sample ID | `Session_ID` |
| Payload input | `clean_payload_list` |
| Gold label | `Tactic` |

不需要先和完整 Step1 prepared metadata join。Golden CSV 本身就包含 payload 與
Tactic，因此可以使用完整 2,000 reviewed rows，避開先前只 match 到部分 Session IDs
的問題。

### 3.2 Label support filtering

Default：

```yaml
min_class_count: 20
conflicting_payload_policy: exclude
deduplicate_payloads: true
```

低於門檻的 labels 不參與第一輪 feasibility training，並記錄在：

```text
prepared/.../manifest.json
  -> excluded_low_support_counts
```

支援度是在排除衝突並依 payload hash 去重後計算。以目前 CSV 的已知分布，
`Persistence` 與 `Defense Evasion` 會被排除；實際結果以每次 manifest 為準。這不是
宣稱它們不重要，而是它們目前不足以做可靠的 stratified train/val/test 評估。

### 3.3 Normal

`Normal (TA9000)` 沒有 MITRE condition embedding。在本實驗中：

```text
Normal -> all-zero gold gates
```

因此 Normal 不再像原 Step1 experiment 一樣被 semantic loss 忽略，而是明確教導
模型：reviewed Normal payload 不應啟動任何 malicious Tactic condition。

### 3.4 Payload duplicate/conflict

每個 payload text 會計算 SHA-256 `payload_hash`。

- 同 hash 如果有不同 gold labels，default 會把該 hash 的所有 rows 排除。
- 同 hash 且同 label 的重複 rows，default 只保留一筆，避免 duplicate payload
  放大某類 train weight 或讓 test 分數虛高。
- Split 以 payload hash 為 group，確保同一 payload 不會跨 train/val/test。

Manifest 會保存 raw/support counts、衝突 hash/row 數與 duplicate removal 數量。若要
資料稽核時遇到衝突直接停止，可把 `conflicting_payload_policy` 改為 `error`。

### 3.5 Stratified group split

Default split：

```yaml
strategy: stratified_group
train_ratio: 0.6
val_ratio: 0.2
test_ratio: 0.2
```

先以 unique payload hash 分組，再依 Tactic stratify。這可讓小型資料的各 supported
classes 都有 train/val/test support，同時避免 duplicate leakage。

這不是 chronological split，所以結果只能稱為 golden-only feasibility，不能稱為
future-time generalization。

### 3.6 Payload embedding cache

Default payload embedder：

```text
nomic-ai/modernbert-embed-base
dimension: 768
normalize: true
overflow: chunk_mean
```

Prepared cache：

```text
outputs/golden_oracle_cvae_step2/prepared/golden_modernbert/
├── x.npy
├── metadata.csv
└── manifest.json
```

Fingerprint 包含 source path/size/mtime、欄位、support threshold 與 embedder config。
只有 fingerprint 相同且三個檔案都存在時才重用。

### 3.7 Standardization

Split 完成後，只使用 train embeddings fit `StandardScaler`，再 transform 全資料。
Scaler 與 standardized matrix 寫入每次 run 的 `scalers/`。Val/test 不參與 scaler
fitting。

---

## 4. Condition 建立

Supported malicious Tactics 會從既有 MITRE condition definition 取得：

```text
experiments/disentangled_cvae_step1/conditions/mitre_attack_v11_3_step1.yaml
```

每個 condition text 由 tactic keywords 與 top-level technique names 組成，使用和
payload 相同的 ModernBERT 產生 768 維 embedding。

Condition geometry 仍使用：

```text
raw condition embedding
  -> subtract condition centroid
  -> optional principal-component removal
  -> row normalization
```

Raw 與 transformed condition cosine matrices/plots 都會輸出。這能檢查 MITRE 共通
背景是否讓 conditions collapse，但 geometry 展開本身不代表 payload 已對齊 Tactic。

---

## 5. 模型架構

### 5.1 為什麼使用 compact architecture

完整 Step1 的舊架構會把固定 `flatten(C_all)` 串到每筆 encoder input，並把
`flatten(gates * C_all)` 串到 decoder。因為 C 對所有 samples 固定，這會產生非常
大的第一層，容易 overfit。

本前導實驗改為：

```text
Encoder input: x only
Decoder input: concat(H, condition_summary)
```

Condition semantics 仍存在於 cosine gate 與 condition summary，但不再重複輸入整張
condition table。

### 5.2 Residual encoder

```text
x -> MLP -> mu_H, logvar_H
H = mu_H + epsilon * exp(0.5 * logvar_H)
```

Default residual dimension：32。

目標：保存 payload 中無法由 Tactic condition 解釋、但 reconstruction 仍需要的
shared/residual information。

### 5.3 Payload behavior projector

```text
q = normalize(Projector(x))
logit_i = cosine(q, C_i) / temperature
predicted_gate_i = sigmoid(logit_i)
```

用途：把 standardized payload embedding 映射到 transformed condition geometry。

### 5.4 Gold gates

對 malicious sample：

```text
gold_gate = one_hot(gold Tactic)
```

對 Normal：

```text
gold_gate = all zeros
```

目前 golden data 是 single-label，因此 gold gate 不會同時啟動多個 conditions。

### 5.5 Condition summary

```text
condition_summary = gates @ condition_matrix
```

Oracle 使用 gold gates；Predicted model 使用 predicted sigmoid gates。

### 5.6 Decoder

```text
concat(H, condition_summary)
  -> MLP decoder
  -> reconstructed standardized payload embedding
```

如果 condition 有效，full reconstruction 應優於使用同一 H、但 condition summary
設為 zero 的 reconstruction。

### 5.7 Plain classifier

```text
x -> MLP -> class logits
```

使用 inverse-frequency class weights，降低多數類別完全支配 objective 的風險。

---

## 6. Loss functions

### 6.1 Oracle CVAE

```text
L_oracle =
    w_rec * MSE(x_hat_gold, x)
  + w_kl * KL(H || N(0, I))
  + w_use * ReLU(margin - (MSE_H_only - MSE_gold))
```

Oracle 沒有 gate prediction loss，因為它的用途就是測 gold-condition upper bound。

### 6.2 Predicted-Gate CVAE

```text
L_predicted =
    w_rec * MSE(x_hat_predicted, x)
  + w_kl * KL(H || N(0, I))
  + w_gate * BCEWithLogits(predicted_gate_logits, gold_gates)
  + w_use * ReLU(margin - (MSE_H_only - MSE_predicted))
```

Gate BCE 使用 train split 算出的 capped positive weights，協助 rare supported Tactics。
Normal 的 target 是所有 gates 為 0，因此也會對 gate BCE 產生 supervision。

### 6.3 Plain classifier

```text
L_classifier = weighted CrossEntropy(class_logits, gold class)
```

### 6.4 Default weights

```yaml
reconstruction: 1.0
kl: 0.01
gate_supervision: 1.0
condition_use: 0.1
condition_use_margin: 0.05
```

KL 使用較小權重是因為此實驗先驗證 condition pathway，不先強迫高度 compressed H。
如果 Oracle/predicted pathway成立，再逐步增加 bottleneck 或 leakage regularization。

---

## 7. Training 流程

三個模型使用完全相同的 train/val/test rows：

1. Fit train-only scaler。
2. 建立 supported malicious conditions 與 Normal class。
3. 訓練 Oracle CVAE。
4. 訓練 Predicted-Gate CVAE。
5. 訓練 Plain Classifier。
6. 每個模型用自己的 `val_loss` early stopping。
7. 重新載入各自 best checkpoint。
8. 在相同 test split 執行 classification、reconstruction 與 controls。

Default optimizer：AdamW，learning rate `3e-4`，weight decay `1e-4`。模型不使用
BatchNorm，降低小資料與 train/validation distribution statistics 不穩定的問題。

---

## 8. 評估方式

### 8.1 Classification metrics

分別輸出：

- Plain classifier
- Predicted-gate model
- Train-majority baseline

Metrics：

- accuracy
- balanced accuracy
- macro F1
- weighted F1
- per-class precision/recall/F1
- normalized confusion matrices

主要看 macro F1 與 per-class recall，不可只看 accuracy。

### 8.2 Oracle reconstruction controls

```text
oracle_zero_gain = MSE_zero - MSE_gold
oracle_shuffled_gain = MSE_shuffled - MSE_gold
```

期望兩者都為正：

- `oracle_zero_gain > 0`：gold condition 優於沒有 condition。
- `oracle_shuffled_gain > 0`：gold condition 優於錯誤 sample 的 condition。

若 Oracle 不通過，代表 conditional decoder/representation 本身尚無證據有效，不應
急著擴大完整 Step1 training。

### 8.3 Predicted-condition utility

```text
predicted_condition_gain = MSE_predicted_zero - MSE_predicted_gate
```

正值表示 predicted gates 對 reconstruction 有實際貢獻。

### 8.4 Oracle 與 predicted 的差距

```text
Oracle gold MSE << Predicted-gate MSE
```

若 Oracle 很好但 predicted 很差，表示 decoder 可以使用 condition，但
payload-to-condition alignment 尚未學好。

### 8.5 Plain classifier 的角色

- Plain classifier 好、predicted gate 差：gate geometry/objective 需要改善。
- 兩者都差：payload embeddings、labels、support 或 split 可能不足。
- Predicted gate 接近 classifier，且 condition gain > 0：CVAE pathway具備進入完整
  Step1 前導價值。

---

## 9. Go / No-Go 判定

`metrics/model_comparison.json` 會產生基本 decision flags：

```text
oracle_beats_zero
oracle_beats_shuffled
predicted_beats_majority_macro_f1
predicted_condition_is_used
```

第一輪 Go 條件：

- [ ] Oracle gold MSE 低於 zero MSE。
- [ ] Oracle gold MSE 低於 shuffled MSE。
- [ ] Predicted-gate macro F1 高於 majority baseline。
- [ ] Predicted condition gain 為正。
- [ ] Predicted gates 沒有全 0、全 1 或 constant collapse。
- [ ] Supported classes 都有 test support。
- [ ] 至少三個 seeds 的方向一致。

更強的 Go 條件：

- Predicted-gate macro F1 接近 plain classifier。
- Rare supported classes 不是全部 recall 0。
- Oracle/predicted gains 有穩定 margin，而不只是浮點數微小正值。
- Shuffled controls 在不同 seeds 仍顯著差於 gold。

如果只有 Oracle 通過，只能結論「conditional pathway upper bound 可行」，還不能開始
大量自動 gate 推論。若 Oracle 與 Predicted 都通過，才有理由擴大標記與完整 Step1
training。

---

## 10. 執行方式

所有命令從 repository root 執行。

### 10.1 Tests

```powershell
uv run python -m unittest discover `
  -s experiments\golden_oracle_cvae_step2\tests -v
```

### 10.2 Prepare only

```powershell
uv run python experiments\golden_oracle_cvae_step2\run_experiment.py `
  --config experiments\golden_oracle_cvae_step2\configs\default.yaml `
  --stage prepare
```

### 10.3 Train only，重用 prepared embeddings

```powershell
uv run python experiments\golden_oracle_cvae_step2\run_experiment.py `
  --config experiments\golden_oracle_cvae_step2\configs\default.yaml `
  --stage train
```

### 10.4 Prepare + train

```powershell
uv run python experiments\golden_oracle_cvae_step2\run_experiment.py `
  --config experiments\golden_oracle_cvae_step2\configs\default.yaml `
  --stage all
```

### 10.5 明確重建 cache

```powershell
uv run python experiments\golden_oracle_cvae_step2\run_experiment.py `
  --config experiments\golden_oracle_cvae_step2\configs\default.yaml `
  --stage prepare `
  --force-prepare
```

只有 source/config 已變更或 cache 不完整時才使用 `--force-prepare`。

### 10.6 觀察 log

```powershell
$run = Get-ChildItem outputs\golden_oracle_cvae_step2 -Directory |
  Where-Object Name -ne "prepared" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Get-Content "$($run.FullName)\logs\experiment.log" -Wait
```

Log 會分別顯示 `oracle`、`predicted`、`classifier` 的 train/val loss、reconstruction、
H-only MSE、condition gain 與 stale epochs。

---

## 11. Outputs

每次 run：

```text
outputs/golden_oracle_cvae_step2/<timestamp>/
├── config_resolved.yaml
├── environment.json
├── logs/
│   └── experiment.log
├── checkpoints/
│   ├── oracle_cvae.pt
│   ├── predicted_gate_cvae.pt
│   └── payload_classifier.pt
├── scalers/
│   ├── scaler.pkl
│   └── x_standard.npy
├── embeddings/
│   ├── condition_embeddings.npz
│   └── condition_embeddings_metadata.json
├── metrics/
│   ├── split_assignments.csv
│   ├── split_summary.json
│   ├── training_history.csv
│   ├── training_history_oracle.csv
│   ├── training_history_predicted_gate.csv
│   ├── training_history_classifier.csv
│   ├── training_summary.json
│   ├── model_comparison.json
│   ├── loss_summary.json
│   ├── behavior_alignment_metrics.json
│   ├── testset_condition_predictions.csv
│   ├── testset_subset_100.csv
│   ├── condition_gate_summary.csv
│   ├── condition_ablation_delta_mse_summary.csv
│   ├── condition_raw_cosine_similarity.csv
│   └── condition_cosine_similarity.csv
├── plots/
│   ├── training_reconstruction_losses.png
│   ├── confusion_matrices.png
│   ├── condition_raw_cosine_similarity.png
│   └── condition_cosine_similarity.png
└── reports/
    └── report.md
```

主要先查看：

```text
metrics/model_comparison.json
reports/report.md
plots/confusion_matrices.png
metrics/testset_condition_predictions.csv
```

`testset_condition_predictions.csv` 包含每筆 test payload 的 gold label、plain
classifier prediction、predicted-gate prediction、各 condition gate、Oracle gold/
zero/shuffled MSE 與 predicted/zero MSE，方便逐筆檢查。

---

## 12. 程式結構

```text
golden_oracle_cvae_step2/
├── configs/default.yaml
├── data.py             # golden prepare/cache/support filtering/group split/scaling
├── model.py            # compact CVAE, predicted gates, plain classifier
├── training.py         # 三模型 training、loss、early stopping、checkpoint
├── evaluate.py         # classification metrics、history/confusion plots
├── run_experiment.py   # prepare/train/all orchestration、controls、reports
├── README.md
└── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_integration.py
```

---

## 13. 已知限制與後續

- Stratified split 不是 chronological generalization。
- 目前 gold labels 是 single-label，不能驗證真實 multi-Tactic amounts。
- Gate threshold `0.5` 尚未 calibration。
- Shuffled control 測試錯誤 condition assignment，但尚未加入 random condition
  embeddings / learned label-ID control；如果需要證明 MITRE text semantics 本身有效，
  這是下一個重要對照。
- H leakage 尚未使用 frozen nonlinear post-hoc probe。
- Oracle condition-use constraint 仍使用 zero-condition intervention；未來可加入
  condition dropout。
- 類別不平衡與低 support labels 仍需更多人工 review。

本實驗通過後的下一步：

1. 對 3 個以上 seeds 重複實驗並彙整 mean ± std。
2. 加入 random condition embedding 與 learned class-ID controls。
3. 加入 calibration、macro-AUPRC 與 H leakage probe。
4. 補標 rare/unsupported Tactics 與 chronological Step1 test window。
5. 將通過的 compact architecture 與 weights 移植到完整 Step1 experiment。
