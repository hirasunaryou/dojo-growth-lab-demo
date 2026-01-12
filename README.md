# DoJo Growth Lab (Interactive demo)

N-F-D-V-R を動かすと「成長の速度・上限」がどう変わるかを、可視化で理解するためのデモです。
数式と意味を固定で表示し、**非エンジニアにも説明できる**ことを目的にしています。
比較は「何もしない（No system）→ 訓練導入（Training）→ DoJo最適化（Optimized）」の
3本線で示し、**10年スケール（520週）**でゆっくりした成長の差を見せる設計です。

## モデルの概要（数式）
**成長モデル**

- dS/dt = v × (Smax − S)
- v = v0 × (N×F×D×V×R)^(1/5)

**Frontier Unlock（上限の拡張）**

- u_rate = u0 × (V×D)
- U_t = U_{t-1} + u_rate × (1 − U_{t-1}/Umax)
- Smax(t) = Smax_base + ΔSmax × (V×D)^gamma × (U_t/Umax)
- Smax_cap = Smax_base + ΔSmax × (V×D)^gamma（理論上限）

## 各パラメータの日本語定義
**N（試行回数 / Trials）**
- 練習ループの回転数。AIロールプレイ等で増やせる
- N↑ → 伸び始めが速くなる

**F（フィードバック密度 / Feedback）**
- 即時・定量・ブレないFB（ルーブリック採点等）で増やせる
- F↑ → 伸びの角度が安定して上がる

**D（深さ / Depth）**
- 難易度/複雑性。背伸び可能な領域を狙う
- D↑ → 学習の上限（Smax）が押し上がる

**V（多様性 / Variety）**
- ケース分布の広さ。長尾・未知パターン・環境変化を含める
- V↑ → Frontier Unlock が進み、Smax が上がりやすい

**R（反復 / Repetition）**
- 弱点に戻って再挑戦できる設計（復習タイミング・再出題）
- R↑ → 学習が収束しやすくなる

## よくある誤解（正解集ではない）
- **「ベンチマーク＝正解」ではない。**  
  ベンチマークは「更新される課題定義＋評価軸」を示すもので、学習設計の更新が前提です。
- **「Nだけ上げれば良い」ではない。**  
  V×D の組み合わせが Frontier Unlock を通じて上限（Smax）を押し上げます。
- **「数式が唯一の真実」ではない。**  
  現場のデータで v0 や u0 を再推定し、学習設計に落とし込むための思考補助です。

## Option A (recommended): Streamlit (web UI)
1) Install deps:
   ```bash
   pip install -r requirements_streamlit.txt
   ```
2) Run:
   ```bash
   streamlit run growth_lab_streamlit.py
   ```

## Option B (no extra deps): Matplotlib sliders (desktop window)
1) Install deps:
   ```bash
   pip install -r requirements_basic.txt
   ```
2) Run:
   ```bash
   python growth_lab_matplotlib_sliders.py
   ```

## Option C: GIF output (PPT向け)
1) Install deps:
   ```bash
   pip install -r requirements_basic.txt
   ```
2) Generate a single sweep:
   ```bash
   python scripts/export_gifs.py --preset optimized --param N --min 0.2 --max 0.9 --frames 30 --fps 10 --out out/gifs
   ```
3) Generate all parameters at once:
   ```bash
   python scripts/export_gifs.py --all --preset optimized --frames 30 --fps 10 --out out/gifs
   ```
4) Compare 3 lines (No system / Training / Optimized) in one GIF:
   ```bash
   python scripts/export_gifs.py --compare3 --preset optimized --param V --frames 30 --fps 10 --out out/gifs
   ```

## Notes
- This is a *conceptual* simulator for stakeholder alignment.
- Calibrate parameters with real data later (e.g., onboarding learning curves, QA scores, KPI deltas).
