# Loss Functions — `splatfacto_on_mesh_uc`

Mesh-guided Gaussian Splatting 的完整損失函數整理，供論文撰寫使用。
公式與 `splat/nerfstudio/models/splatfacto_on_mesh_uc.py` 的實作逐項對應
（行號為 2026-08-06 當下狀態，函式名較行號穩健）。

---

## 1. 符號約定

| 符號 | 意義 |
|---|---|
| $\mathcal{G}=\{1,\dots,N\}$ | base + detail Gaussian，綁定於 mesh 面上 |
| $\pi(i)\in\{1,\dots,F\}$ | Gaussian $i$ 所屬的三角面（`gaussians_to_mesh_indices`） |
| $\mathcal{V}=\{1,\dots,V\}$ | 頂點填補層 Gaussian（一頂點一顆，位置固定於 $\mathbf{x}_v$，不參與訓練位移） |
| $F$, $\mathbf{b}_f$ | mesh 面數、面 $f$ 的重心 |
| $\alpha_i$ | **渲染** opacity，已套用 base-layer floor（`opacities` property） |
| $\tilde\alpha_i=\sigma(o_i)$ | **原始** opacity，未套 floor（直接對參數取 sigmoid） |
| $\alpha^{\mathrm{v}}_v$ | 頂點層 Gaussian 的 opacity |
| $\mathbf{c}_i\in\mathbb{R}^3$ | 球諧函數的 DC 係數（顏色主項，不含視角相關項） |
| $\mathbf{u}_i,\mathbf{u}_v$ | 面局部 2D 平面座標；Gaussian 中心已夾至自身三角形內 |
| $s_{i,x},s_{i,y},s_{i,z}$ | Gaussian 三軸尺度（$z$ 為面法向厚度） |
| $\sigma_i=\min(s_{i,x},s_{i,y})$ | 保守等向半徑，取面內兩軸較小者 |
| $I,\hat I$ | 真實影像、渲染影像 |
| $A,\hat D$ | accumulation map、渲染深度圖 |
| $t$, $t_{\text{stop}}$ | 目前訓練步、停止緻密化的步數 |

---

## 2. 總損失

$$
\mathcal{L}=\mathcal{L}_{\text{photo}}+\mathcal{L}_{\text{scale}}+\mathcal{L}_{\text{opa}}+\mathcal{L}_{\text{cov}}+\mathcal{L}_{\text{color}}
$$

各項權重已內含於下列定義。實作上以 dict 回傳後由 trainer 直接加總
（`get_loss_dict`, L3977）。

---

## 3. 光度項 $\mathcal{L}_{\text{photo}}$

$$
\mathcal{L}_{\text{photo}}
=(1-\lambda_{\text{ssim}})\,\mathcal{L}_{1}
+\lambda_{\text{ssim}}\bigl(1-\mathrm{SSIM}(I,\hat I)\bigr)
+\lambda_{\text{acm}}\mathcal{L}_{\text{acm}}
+\lambda_{\text{d}}\mathcal{L}_{\text{depth}}
$$

### 3.1 掠角加權 L1

$$
\mathcal{L}_{1}=\frac{\sum_{p}w_p\,\lVert I_p-\hat I_p\rVert_1}{\sum_{p}w_p},
\qquad
w_p=\bigl(\max(\cos\theta_p,\,w_{\min})\bigr)^{\gamma}
$$

$\cos\theta_p$ 為像素 $p$ 處表面法向與視線方向夾角的餘弦。掠角觀測數量少、
幾何資訊最不可靠，全額計入會讓模型被少數不良觀測拉走；下限 $w_{\min}$
確保這些像素仍保有梯度。停用時退化為 $\mathcal{L}_1=\mathrm{mean}\lvert I-\hat I\rvert$。

### 3.2 累積透明度 $\mathcal{L}_{\text{acm}}$

$$
\mathcal{L}_{\text{acm}}=\frac{1}{|\mathcal{P}^{-}|}\sum_{p\in\mathcal{P}^{-}}\bigl(\tau_A-A_p\bigr),
\qquad
\mathcal{P}^{-}=\{p:A_p<\tau_A\}
$$

懲罰累積不透明度未達 $\tau_A$ 的像素，迫使模型填實不該透出背景的區域。
**平均僅取未達標像素**，使損失強度不隨未達標像素比例下降而稀釋。

停止條件：$t\ge t_{\text{stop}}$ 時強制 $\lambda_{\text{acm}}\leftarrow 0$。
緻密化停止後只剩剔除機制在收斂點數，此時續用本項會持續推高 opacity，
使冗餘 Gaussian 無法降至剔除門檻以下，點數永遠無法穩定。

### 3.3 深度 $\mathcal{L}_{\text{depth}}$

$$
\mathcal{L}_{\text{depth}}=\frac{1}{|\mathcal{P}|}\sum_{p}\bigl\lvert D^{\text{mesh}}_p-\hat D_p\bigr\rvert
$$

以 guide mesh 的深度圖監督渲染深度。需 dataparser 提供 `mesh_depth`，
否則此項恆為 0。

---

## 4. 形狀各向異性 $\mathcal{L}_{\text{scale}}$

$$
\mathcal{L}_{\text{scale}}=\frac{1}{N}\sum_{i}
\left[\max\!\left(\frac{\max(s_{i,x},s_{i,y})}{\min(s_{i,x},s_{i,y})},\,r_{\max}\right)-r_{\max}\right]
$$

僅在面內長寬比超過 $r_{\max}$ 時產生梯度，防止 Gaussian 退化成針狀。

**僅約束面內兩軸 $(x,y)$**：法向厚度 $s_{i,z}$ 是刻意壓扁的設計
（`face_flat_coef`），納入比值會變成持續懲罰模型自身的預期形狀。

---

## 5. Opacity 二值化 $\mathcal{L}_{\text{opa}}$

$$
\mathcal{L}_{\text{opa}}=\frac{\lambda_{\text{opa}}}{N}\sum_{i}\tilde\alpha_i\,(1-\tilde\alpha_i)
$$

$x(1-x)$ 於 $x=0.5$ 取最大、於 $x\in\{0,1\}$ 為零，因此將每顆 Gaussian
推向全透明或全不透明，抑制大量半透明殘影疊加。

> **實作註記**：本項使用**原始** opacity $\tilde\alpha$，未經 base-layer floor。
> 若論文中將本項描述為「不作用於 base layer」，實作需補上對應的排除項。

---

## 6. 幾何覆蓋率 $\mathcal{L}_{\text{cov}}$

核心概念：在 mesh 表面上取一組**檢查點**，量測每個檢查點實際接收到多少
Gaussian 密度，密度不足者施以懲罰（`_compute_coverage_density`, L3872）。

檢查點共兩類，合計 $V+F$ 個：

| 檢查點 | 數量 | 密度符號 | 意義 |
|---|---|---|---|
| 每個 mesh 頂點 $v$ | $V$ | $\rho_v$ | 三角形的**角落**是否被蓋到 |
| 每個面的重心 $\mathbf{b}_f$ | $F$ | $\rho_f$ | 三角形的**內部**是否被蓋到 |

兩者分開檢查，是為了讓損失具備位置感知能力：Gaussian 全數堆積於面中心時
$\rho_f$ 可以很高，但 $\rho_v$ 仍會偏低而產生梯度。

### 6.1 本節專用符號

| 符號 | 意義 |
|---|---|
| $\mathcal{N}(v)$ | 所有「所屬面以 $v$ 為頂點之一」的 Gaussian 集合，即 $\{i:\,v\in\text{verts}(\pi(i))\}$ |
| $\text{verts}(f)$ | 面 $f$ 的三個頂點 |
| $\mathbf{u}^{(f)}_{\ast}$ | 點 $\ast$ 在**面 $f$ 自身局部 2D 平面座標系**中的座標 |
| $\mathbf{x}_{\ast}$ | 點 $\ast$ 的 3D 世界座標 |
| $\sigma_i=\min(s_{i,x},s_{i,y})$ | Gaussian $i$ 的保守等向半徑 |
| $\sigma^{\mathrm{v}}_v$ | 頂點層 Gaussian $v$ 的對應半徑 |

> 局部座標系依附於**面**：同一個頂點 $v$ 被多個面共享，在每個面的座標系中
> 座標不同。故下式中的 $\mathbf{u}^{(\pi(i))}_{v}$ 必須標明是在
> Gaussian $i$ 所屬那個面的座標系裡取值——距離是在該 Gaussian 自己的面平面上
> 計算的，而非某個全域平面。

### 6.2 頂點檢查點的覆蓋密度

$$
\rho_v=
\underbrace{\sum_{i\in\mathcal{N}(v)}
\overbrace{\alpha_i}^{\text{(A)}}
\underbrace{\exp\!\left(-\frac{\lVert\mathbf{u}^{(\pi(i))}_{i}-\mathbf{u}^{(\pi(i))}_{v}\rVert^{2}}
{2\sigma_i^{2}}\right)}_{\text{(B)}}}_{\text{(I) base + detail 層}}
\;+\;\underbrace{\alpha^{\mathrm{v}}_v}_{\text{(II) 頂點層}}
$$

| 標記 | 代表什麼 |
|---|---|
| **(A)** $\alpha_i$ | Gaussian $i$ 的**渲染** opacity。愈不透明，貢獻的覆蓋愈多 |
| **(B)** 指數項 | 高斯核衰減：Gaussian 中心離檢查點愈遠、或自身愈小（$\sigma_i$ 愈小），貢獻愈少。中心正好落在檢查點時取 $\exp(0)=1$ |
| **(I)** 求和 | 對**所有以 $v$ 為角落的面上的 Gaussian** 累加。一個頂點被多個面共享，任一相鄰面的 Gaussian 蓋到它都算數 |
| **(II)** | 頂點填補層的貢獻。該層 Gaussian 的位置依構造**恰為** $\mathbf{x}_v$，距離為 0，故 (B) 化簡為 1，整項退化成它自己的 opacity |

### 6.3 面重心檢查點的覆蓋密度

$$
\rho_f=
\underbrace{\sum_{i:\,\pi(i)=f}\alpha_i
\exp\!\left(-\frac{\lVert\mathbf{u}^{(f)}_{i}-\mathbf{u}^{(f)}_{\mathbf{b}_f}\rVert^{2}}
{2\sigma_i^{2}}\right)}_{\text{(III) 本面自己的 base + detail}}
\;+\;
\underbrace{\sum_{v\in\text{verts}(f)}\alpha^{\mathrm{v}}_v
\exp\!\left(-\frac{\lVert\mathbf{x}_v-\mathbf{x}_{\mathbf{b}_f}\rVert^{2}}
{2(\sigma^{\mathrm{v}}_v)^{2}}\right)}_{\text{(IV) 三個角落的頂點層交叉貢獻}}
$$

| 標記 | 代表什麼 |
|---|---|
| **(III)** | 只累加**歸屬於本面**的 Gaussian（$\pi(i)=f$）。與 (I) 不同：頂點是共享的，重心不是 |
| **(IV)** | 本面三個頂點上的頂點層 Gaussian，延伸進來蓋到本面重心的部分。距離採 **3D 世界距離**而非面局部 2D——頂點由多個面共享，沒有單一面平面可投影；而兩點皆位於表面上，世界距離已是表面距離的良好近似 |

### 6.4 損失

$$
\mathcal{L}_{\text{cov}}=\lambda_{\text{cov}}\left[
\underbrace{\frac{1}{V}\sum_{v=1}^{V}\bigl(\tau_c-\rho_v\bigr)_{+}}_{\text{角落缺口}}
+\underbrace{\frac{1}{F}\sum_{f=1}^{F}\bigl(\tau_c-\rho_f\bigr)_{+}}_{\text{內部缺口}}\right]
$$

| 符號 | 意義 |
|---|---|
| $(\cdot)_{+}=\max(0,\cdot)$ | 單邊懲罰：密度**超過**目標不給獎勵也不罰，只罰不足的部分 |
| $\tau_c$ | 覆蓋目標（`coverage_target`）。密度達到即停止施力 |
| $\lambda_{\text{cov}}$ | 整體權重（`coverage_lambda`） |

### 6.5 $\tau_c$ 的尺度意義

密度是**加總**而非取最大值（實作為 `scatter_add`），因此 $\rho$ 可以超過 1：
多顆 Gaussian 疊在同一檢查點時密度會累加。

基準點：**單獨一顆完全不透明（$\alpha=1$）、中心正好落在檢查點上的 Gaussian，
貢獻恰為 $1.0$**。因此

- $\tau_c=1.0$ 要求「每個檢查點至少達到相當於一顆完全不透明 Gaussian 正中的密度」
- $\tau_c=0.6$ 則等於明確容忍「蓋到六成就不再施力」

這使 $\tau_c$ 具有可解釋的物理尺度，而非任意的無因次門檻。

### 6.6 性質與限制

- **位置感知**：不同於「每面足跡總面積」式的預算約束，Gaussian 堆積於中心
  無法滿足角落檢查點。
- **與相機無關**：純幾何量，故亦能約束光度損失永遠觀測不到的遮蔽區域。
- **保守等向半徑**：採 $\sigma_i=\min(s_{i,x},s_{i,y})$ 而非各軸實際尺度，
  避免高度各向異性的 Gaussian 以單一方向的延展偽造覆蓋。
- **只能改變既有 Gaussian，無法生成新的**：本項的梯度只能移動或放大既有
  Gaussian。新點的生成由 `coverage_densify_scale`（依缺口提高緻密化觸發機率）
  與 `coverage_rescue_thresh`（低於門檻即無條件補點）負責，兩者與本項共用
  同一密度函數 $\rho$，但屬於緻密化策略而非損失項。

---

## 7. 顏色一致性 $\mathcal{L}_{\text{color}}$

抑制空間重疊 Gaussian 之間的顏色歧異。閃爍量級正比於
$\alpha_1\alpha_2\lVert\Delta\mathbf{c}\rVert$（兩顆近乎共面的半透明 Gaussian
深度排序隨視角翻轉時的可見色跳），本項直接壓制 $\lVert\Delta\mathbf{c}\rVert$
因子，**不改變 Gaussian 的數量、尺寸或位置，因此不付出覆蓋率代價**
（`_compute_color_consistency`, L3820 附近）。

以每面 opacity 加權平均色為中介，而非逐對計算——最小化各成員對共同平均的
平方差，等價於最小化其兩兩平方差（差一常數因子），複雜度由需要逐步空間鄰居
搜尋降為 $O(N+E)$ 的 scatter 運算：

$$
W_f=\!\!\sum_{i:\pi(i)=f}\!\!\alpha_i,
\qquad
\bar{\mathbf{c}}_f=\frac{1}{W_f}\!\!\sum_{i:\pi(i)=f}\!\!\alpha_i\mathbf{c}_i
$$

$$
\mathcal{L}_{\text{color}}=\lambda_{\text{col}}
\Bigl(\mathcal{L}^{\text{in}}+\mathcal{L}^{\text{adj}}+\mathcal{L}^{\text{fold}}\Bigr)
$$

**面內項**：

$$
\mathcal{L}^{\text{in}}=\frac{\sum_i\alpha_i\lVert\mathbf{c}_i-\bar{\mathbf{c}}_{\pi(i)}\rVert^{2}}{\sum_i\alpha_i}
$$

**面間項**（$\mathcal{S}$ 代入兩種面對集合）：

$$
\mathcal{L}^{\mathcal{S}}=\frac{\sum_{(a,b)\in\mathcal{S}}W_aW_b\lVert\bar{\mathbf{c}}_a-\bar{\mathbf{c}}_b\rVert^{2}}
{\sum_{(a,b)\in\mathcal{S}}W_aW_b}
$$

- $\mathcal{S}=\mathcal{E}_{\text{adj}}$：**共頂點**鄰接面對（`face_adjacency_pairs`, `share="vertex"`）
- $\mathcal{S}=\mathcal{E}_{\text{fold}}$：幾何鄰近但拓撲遙遠的摺疊面對（`_set_face_overlap_pairs`）

**三項各自正規化**（不併為單一面對集合）：摺疊面對數量為 $10^3$ 量級，
而共頂點面對達 $6.3\times10^5$，共用分母會使摺疊項被稀釋至無效。

**設計要點**：

- 鄰接採**共頂點**而非共邊。一個三角形有 3 個共邊鄰面但約 12 個共頂點鄰面，
  而 base Gaussian 的足跡約為自身三角形面積的 $2.6\text{–}2.8$ 倍，
  故角落鄰面在物理上的重疊程度與共邊鄰面相當。
- 所有加權係數 $\alpha$、$W$ 皆 **detach**：梯度只流向顏色。本項的職責是使顏色
  趨於一致，而非讓模型藉由壓低 opacity 規避懲罰（opacity 的取捨由
  $\mathcal{L}_{\text{opa}}$ 與覆蓋率機制負責）。
- 僅約束 DC 項，不約束高階球諧係數，以保留真實的視角相關變化。
- 頂點填補層排除在外（不受 $\pi(\cdot)$ 索引，且其 opacity 實測收斂至
  $\sim\!0.007$，對 $\alpha_1\alpha_2$ 貢獻可忽略）。
- 無 Gaussian 的面其 $\bar{\mathbf{c}}_f=\mathbf{0}$ 並非真實顏色；以
  $W_aW_b$ 加權使該類面對貢獻趨近 0，而非把鄰面拉向黑色。

---

## 8. 超參數

| 符號 | 參數名 | 目前值 |
|---|---|---|
| $\lambda_{\text{ssim}}$ | `ssim_lambda` | 0.2 |
| $w_{\min}$ | `grazing_weight_floor` | 0.1 |
| $\gamma$ | `grazing_weight_power` | 2.0 |
| $\lambda_{\text{acm}}$ | `acm_lambda` | **0（停用）** |
| $\tau_A$ | （硬編碼） | 0.95 |
| $t_{\text{stop}}$ | `stop_split_at` | 25000 |
| $\lambda_{\text{d}}$ | `mesh_depth_lambda` | **0（停用）** |
| $r_{\max}$ | `max_gauss_ratio` | 2.0 |
| $\lambda_{\text{opa}}$ | `opacity_reg_lambda` | 0.05 |
| $\lambda_{\text{cov}}$ | `coverage_lambda` | 1.0 |
| $\tau_c$ | `coverage_target` | 1.0 |
| $\lambda_{\text{col}}$ | `color_consistency_lambda` | 0.05 |

---

## 9. 撰寫時的注意事項

1. **$\alpha$ 與 $\tilde\alpha$ 的不一致**：$\mathcal{L}_{\text{cov}}$ 與
   $\mathcal{L}_{\text{color}}$ 使用**渲染** opacity（含 base floor，衡量實際
   渲染結果，屬刻意設計）；$\mathcal{L}_{\text{opa}}$ 使用**原始** opacity。
   論文中若不擬討論此差異，可統一以 $\alpha$ 表示並於實作附錄說明。

2. **$\mathcal{L}_{\text{acm}}$ 與 $\mathcal{L}_{\text{depth}}$ 權重目前為 0**。
   若論文描述的是最終使用配置，建議省略或明確標註為「本實驗停用」；
   若要保留於方法章節，需一併提供消融設定說明。

3. **$\mathcal{L}_{\text{cov}}$ 與 $\mathcal{L}_{\text{color}}$ 是本方法相對
   vanilla 3DGS 的主要新增項**，前者提供覆蓋率的幾何保證，後者處理由此
   保證所必然引入的重疊閃爍。兩者的張力（覆蓋率與閃爍為同一槓桿的兩端）
   值得在論文中明確論述。

4. 覆蓋率的**生成**機制（`coverage_densify_scale`、`coverage_rescue_thresh`）
   與 base / vertex 填補層屬於架構與緻密化策略，非損失項，宜於方法章節
   另立小節，不要混入損失函數的表述。
