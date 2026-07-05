# Condition 解偶改善 TODO

本文記錄後續要如何改善 `experiments/disentangled_cvae_step1` 的設計，使模型更接近原本期待的 Condition 解偶：不是只讓 condition embedding 幫助 reconstruction，而是讓 gate 能夠更可信地表示「某筆 payload 啟動了哪些惡意行為概念」。

目前設計重點：

- `payload_list` 會被 embed 成 payload embedding `x`。
- condition descriptions 會被 embed 成 condition matrix `C_all`。
- encoder 看 `concat(x, flatten(C_all))`，輸出 residual `H` 和每個 condition 的 `gates`。
- decoder 看 `concat(H, flatten(gates * C_all))` 來 reconstruct `x`。

目前主要風險：

- `C_all` 對每筆 sample 都一樣，encoder 可能只把它當成固定背景資訊或 bias。
- gate 可能只是 reconstruction shortcut，不一定是在做 payload 與 condition 的語意比對。
- 一般 embedding model 不保證 raw/semi-structured payload embedding 與 MITRE tactic description embedding 已經在同一個安全語意空間中對齊。
- reconstruction 做得好，不等於 condition 分解語意正確。

## Step 1. 重新定義 Condition 粒度

### 要修正什麼

目前 condition 主要來自 MITRE tactic-level 描述，例如 Reconnaissance、Execution、Persistence。這種粒度對 payload-level data 可能太抽象。後續應改成或補充更接近 payload 可觀測特徵的 behavior-level conditions。

建議 condition 粒度範例：

- path probing
- directory traversal
- SQL injection attempt
- command injection
- credential login attempt
- webshell upload or webshell access
- known exploit URI access
- scanner-like repeated endpoint discovery
- sensitive file access
- suspicious script or binary download
- protocol misuse
- encoded or obfuscated payload

每個 condition 不應只是一句 MITRE 定義，應補充 payload 觀測層面的描述。

每個 condition 建議包含：

- `label`: 可讀名稱。
- `id`: 穩定識別碼。
- `description_full`: 給 embedding model 的完整描述。
- `observable_patterns`: payload 中常見的可觀測線索。
- `positive_examples`: 典型正例 payload 或 payload 片段。
- `negative_examples`: 容易混淆但不應啟動此 condition 的例子。
- `boundary_notes`: 與其他 condition 的差異。
- `parent_tactic`: 如需保留 MITRE 對應，可把 behavior condition 映射回 tactic。

### 用意

condition embedding `C_i` 要能成為 payload embedding 的可解釋語意錨點。若 condition 太抽象，模型難以從單筆 payload 中拆出 tactic-level intent。

更好的定義應該是：

```text
C_i 代表 payload 中可觀測的惡意行為概念，
而不是只代表高階攻擊目的。
```

### 注意事項

- 不要把 `Sess_Tactic_predict` 直接當成 ground truth condition。它目前只是 metadata，而且是前一步模型預測結果。
- condition 粒度不能太細到每個 URI 或 signature 都是一個 condition，否則會變成 signature matching，不是語意解偶。
- condition 之間要盡量互斥或至少邊界清楚，否則 gate 很容易同時啟動多個高度重疊概念。
- 如果保留 MITRE tactic，建議 tactic 作為上層 taxonomy，不要直接作為唯一 condition set。

### 驗收指標

- 每個 condition 都能被人工指出 payload 中可能對應的 evidence。
- condition description 看起來不像單純抽象安全名詞，而是包含 payload 可觀測行為。
- condition cosine similarity 不應全部高度相似；高度相似的 condition 要合併或補充邊界描述。

## Step 2. 建立 Payload-Condition Explicit Alignment

### 要修正什麼

目前 payload embedding `x` 和 condition embedding `C` 只是使用同一個 embedding model 產生，這只能建立很弱的共享語意空間假設。後續應加入明確的 alignment 訓練或評估資料，使：

```text
sim(payload, correct_condition) 高
sim(payload, wrong_condition) 低
```

可行資料來源：

- 人工 golden review 小樣本。
- 規則標記的 weak labels。
- 已知 exploit/path/signature 對應到 behavior condition 的弱監督資料。
- 現有 metadata label 只作輔助分析，不要直接當強標籤。

建議建立 payload-condition pair dataset：

```text
positive pair:
  (payload with SQL injection behavior, C_sql_injection)

negative pair:
  (payload with SQL injection behavior, C_recon)
  (payload with SQL injection behavior, C_credential_login)
```

模型層面可加入：

- contrastive loss
- InfoNCE loss
- triplet loss
- pairwise ranking loss
- projection head for payload and condition embeddings

### 用意

一般 embedding model 不保證 payload text 與安全概念描述自然對齊。alignment 的目的，是讓 condition embedding 不只是文字語意錨點，而是和 payload-level evidence 有可學習、可驗證的關係。

沒有 alignment 時：

```text
gate 有用，只能表示它幫助 reconstruction。
gate 高，不代表該 condition 語意正確。
```

有 alignment 後：

```text
gate 高，才比較能解釋成 payload 和該 condition 有語意關係。
```

### 注意事項

- weak labels 可以用，但要在 report 中明確標註來源與可信度。
- contrastive negatives 要包含 hard negatives，例如 command injection vs SQL injection、path probing vs sensitive file access。
- 不要只用 random negatives，否則 alignment 任務太簡單，實際解偶仍可能失敗。
- 如果 embedding model 對 payload token 理解很差，可能需要 domain-specific fine-tuning 或 payload-specific encoder。

### 驗收指標

- 在人工 review 或 held-out weak-label set 上，correct condition similarity 應高於 wrong condition。
- top-k retrieved conditions 應和人工可理解的 payload behavior 一致。
- 用真 condition descriptions 應明顯優於 shuffled/random condition descriptions。

## Step 3. 改 Gate 語意：從任意 MLP 輸出改成 Explicit x-C Matching

### 要修正什麼

目前 gate 是由 encoder hidden state 經過 linear layer + sigmoid 產生：

```text
gates = sigmoid(W hidden)
```

這不保證 gate 是 payload 與 condition 的相似度。後續應考慮讓 gate 明確依賴 `x` 和每個 `C_i` 的 matching。

建議方向 1：similarity gate

```text
z_x = payload_projection(x)
z_c_i = condition_projection(C_i)
gate_i = sigmoid(scale * cosine(z_x, z_c_i) + bias_i)
```

建議方向 2：condition-wise scoring MLP

```text
score_i = MLP([z_x, z_c_i, z_x * z_c_i, abs(z_x - z_c_i)])
gate_i = sigmoid(score_i)
```

建議方向 3：cross-attention gate

```text
condition C_i 作為 query
payload token/features 作為 key/value
每個 condition 從 payload 中找 evidence
```

如果仍使用 single vector payload embedding，similarity gate 是較簡單的第一版。若未來能保留 payload token-level features，cross-attention 會更符合「每個 condition 找自己的 evidence」。

### 用意

gate 應該代表：

```text
這筆 payload 和 condition i 的關係強度
```

而不是：

```text
decoder 為了 reconstruct x 所學到的一組任意權重
```

讓 gate 由 x-C matching 產生，可以使 condition 解釋更有根據。

### 注意事項

- gate 使用 sigmoid 時允許 multi-label；適合一筆 payload 同時有多個 behavior。
- gate 使用 softmax 時會強迫 condition 互斥；除非明確假設每筆 payload 只能對應一個 condition，否則不建議一開始就使用純 softmax。
- 可以保留 sparse loss，但要避免 sparse weight 太強導致所有 gates 接近 0。
- 若使用 similarity gate，condition embedding quality 和 projection alignment 會變得非常關鍵。

### 驗收指標

- gate top-k 與 x-C similarity ranking 應有一致性。
- gate 高的 condition 應能在 payload 中找到合理 evidence。
- gate 不應在所有 sample 上呈現幾乎固定的 pattern；否則代表 gate 主要學到 condition prior，而不是 sample-specific behavior。

## Step 4. 限制 H，避免 H 偷藏 Condition 資訊

### 要修正什麼

目前已有 KL、residual constraint、H-only reconstruction 監控，方向正確。後續要更明確定義：

```text
H 只保留 condition 無法解釋的殘差。
gated C pathway 負責 condition-related behavior。
```

可加強的方向：

- 降低 `residual_dim`，測試不同大小對 condition usage 的影響。
- 增強 residual constraint，使 `H-only` 不應接近 full reconstruction。
- 加入 adversarial/probe 檢查，避免從 `H` 輕易預測 condition。
- 監控 `full reconstruction`、`H-only reconstruction`、`C-only reconstruction` 的差距。
- 對 `H` 加更強 bottleneck 或 noise，使它不容易成為完整記憶通道。

建議要追的核心數字：

```text
full_mse
h_only_mse
c_only_mse
h_only_mse - full_mse
condition ablation delta
gate sparsity
condition predictability from H
```

### 用意

如果 `H-only` 幾乎和 full reconstruction 一樣好，代表模型主要靠 H，condition pathway 可能只是裝飾。

理想狀態：

```text
full reconstruction 最好
H-only 明顯變差
C-only 能重建 condition-related 部分，但無法完整重建
高 gate condition 被 ablate 後 reconstruction 明顯變差
H 不容易預測 condition
```

### 注意事項

- 不能把 H 壓太小到 reconstruction 完全崩潰，否則 gate 可能被迫亂開。
- residual constraint 太強可能導致模型犧牲 reconstruction，得到看似 sparse 但不可靠的 gates。
- 要用多個 seed 檢查穩定性，否則可能只是某次初始化造成的分解。
- condition 解偶是 identifiability 問題，只靠 reconstruction 仍不足以保證語意正確。

### 驗收指標

- `full_mse` 明顯低於 `h_only_mse`。
- 高 gate condition 的 ablation delta 明顯高於低 gate condition。
- 從 `H` 訓練簡單 probe 預測 condition 時，效果不應太好；若很好，代表 H 偷藏 condition。
- 同一 payload 在不同 seed 下的 top gate 應大致穩定。

## Step 5. 加入可驗證的 Condition Supervision

### 要修正什麼

完全無監督 reconstruction 很難保證 gate 對應到人類定義的 condition。後續應加入最低限度的 supervision，即使是少量人工或 weak supervision。

可加入的 supervision：

- gate-label BCE loss：multi-label condition target。
- gate ranking loss：positive condition gate > negative condition gate。
- contrastive payload-condition loss：正確 pair 拉近，錯誤 pair 推遠。
- top-k consistency loss：要求人工或 weak label 中的 condition 排在前面。
- calibration/evaluation only labels：若不想直接訓練，也至少用作驗證。

建議資料策略：

1. 先建立小型人工 golden review set。
2. 每筆 payload 標註 0 到多個 behavior-level conditions。
3. 記錄 evidence span 或 evidence note。
4. 將 golden set 分成 alignment train/dev/test，避免只在同一批資料上調參。
5. weak labels 可大量補充，但要和 human labels 分開報告。

### 用意

supervision 的目的不是把模型變成純分類器，而是給 gate 一個語意方向：

```text
reconstruction loss 確保 representation 有資訊。
condition supervision 確保 gate 對應人類定義的 condition。
contrastive alignment 確保 payload 和 condition embedding 在同一空間可比。
```

這樣 gate 才能更接近「condition 解偶」而不是「任意 latent decomposition」。

### 注意事項

- 不要只看 classification accuracy，否則會偏離 disentanglement 目標。
- supervision loss 權重要小心調，太強會讓模型只學 label shortcut，太弱則 gate 仍可能沒有語意。
- 如果使用 weak labels，要防止模型重現 weak label 的錯誤偏見。
- 如果 condition 是 multi-label，不要用單一 softmax CE 當唯一 supervision。
- golden review 的 condition 定義要和 Step 1 的 behavior-level condition 保持一致。

### 驗收指標

- gate top-1/top-k 在 golden review set 上與人工 condition 有一致性。
- positive condition 的 gate 平均高於 hard negative condition。
- 加入 supervision 後，ablation utility 不應消失；否則 gate 可能只是分類頭，不是真的進 reconstruction。
- 真 condition set 應優於 shuffled labels 或 random condition embeddings。

## 建議優先順序

1. 先重寫 condition schema，讓 condition 變成 payload 可觀測的 behavior-level concepts。
2. 建立小型 golden review set，至少能驗證 gate 是否語意正確。
3. 加入 payload-condition alignment，先從 contrastive/ranking objective 開始。
4. 將 gate 改成 explicit x-C matching，而不是只由 encoder MLP 自由產生。
5. 加強 H 的限制與監控，確認 condition pathway 真的承擔可解釋資訊。

## 最重要的判斷標準

後續不要只問：

```text
reconstruction 有沒有變好？
```

還要問：

```text
gate 高的 condition 是否有 payload evidence？
拿掉 gate 高的 condition 是否真的傷害 reconstruction？
H-only 是否明顯比 full 差？
真 condition 是否比 random/shuffled condition 好？
不同 seed 的 gate 是否穩定？
人工 review 是否認同 top gate？
```

只有這些條件逐步成立，才能比較有把握說目前模型真的在做 Condition 解偶。
