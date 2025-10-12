# **Project Proposal: Customer Revenue Uplift for Telco Marketing**

### **1️⃣ Judul**

**“Predict & Simulate: Customer Revenue Uplift for Telco Marketing”**
_(Catchy, profesional, terdengar advanced & scalable)_

---

### **2️⃣ Latar Belakang**

- Telco di Indonesia punya jutaan pelanggan dengan profil, penggunaan, dan perilaku berbeda.
- Tantangan marketing: menargetkan pelanggan yang tepat supaya **revenue naik** tanpa menambah cost.
- Banyak perusahaan masih pakai **rule-based segmentation atau A/B testing manual** → lambat & tidak optimal.
- AI dapat membantu: **predict revenue uplift per pelanggan, segmentasi cerdas, & simulasi skenario campaign**.

**Inspirasi:** Rio Nur Arifin di Telkomsel: fokus marketing DS → ML, segmentation, uplift modeling.
**Tapi kelemahan Rio:** domain terbatas, tools bias, kurang cutting-edge.
**Project ini menambal kelemahan tersebut**:

- Gunakan **advanced ML, Graph ML, generative AI**, plus **interactive dashboard & scenario simulation**.
- Buat **prototype yang measurable**, bukan sekadar demo.

---

### **3️⃣ Problem Statement**

1. Bagaimana memprediksi pelanggan dengan **potensi revenue uplift terbesar** untuk campaign tertentu?
2. Bagaimana memvisualisasikan **simulasi berbagai strategi campaign** (top-tier, mid-tier, mix) untuk melihat expected revenue & cost?
3. Bagaimana menggabungkan **feature explainability & network influence** supaya stakeholder bisa percaya keputusan AI?

---

### **4️⃣ Objective**

- Bangun **predictive ML model** untuk revenue uplift.
- Bangun **interactive dashboard / AI prototype** yang bisa:

  - Simulasi _what-if campaign scenarios_
  - Rank pelanggan berdasarkan expected revenue uplift
  - Visualisasi network influence (Graph ML)
  - Tampilkan top drivers & SHAP explainability

- Gunakan **synthetic/public dataset** yang mirip real telco.
- Hasil harus bisa:

  - Measurable → contoh: “Target top 5% HVC → expected revenue Rp X juta / minggu”
  - Viral → visual interaktif, storytelling jelas, impress profesional AI & software dev

---

### **5️⃣ Dataset**

- **Customer profile:** usia, kota, SES proxy (ARPU/top-up), plan type
- **Usage metrics:** data, voice, SMS, apps usage, site_id KPI
- **Campaign history:** past exposure, conversions, uplift outcome
- **Network info optional:** high UL interference, packet loss (buat insight advanced)
- **Label:** revenue uplift (binary conversion or continuous delta revenue)

> Bisa pakai **public dataset** + generate synthetic tambahan pakai **GAN / LLM** supaya scale & variabilitas mirip real telco.

---

### **6️⃣ Methodology / Tech Stack**

**Step 1 – Data preparation & feature engineering**

- Time-based features: 7/30/90d aggregation
- Interaction features: usage \* network quality
- Target encoding untuk categorical (plan, site, segment)
- Synthetic data generator (optional)

**Step 2 – ML Modeling**

- Baseline: RandomForest, XGBoost, LGBM, CatBoost
- Uplift modeling: T-learner / X-learner (propensity-aware)
- Graph ML: customer influence network (networkx + GNN optional)
- Optional: NLP on customer feedback → sentiment + correlation dengan conversion

**Step 3 – Explainability**

- SHAP / LIME feature importance
- Graph centrality + influence visualization

**Step 4 – Interactive Prototype**

- **Streamlit / Gradio** dashboard
- Interactive scenario selector (top 5%, mid-tier, custom)
- Visualization: revenue uplift projection, cost estimation, top features, network influence graph

**Step 5 – Simulation / What-If Scenarios**

- Simulasi berbagai strategi target → expected revenue / cost / ROI
- Scenario comparison (visual + table + charts)

---

### **7️⃣ Roadmap (4 Weeks Plan)**

**Week 1 – Setup & Data Engineering**

- Environment setup (Python, Streamlit, ML libs, networkx, pandas, SQL/Hive if needed)
- Data prep: generate dummy/synthetic data → validate distributions mirip real telco
- Feature plan & catalog

**Week 2 – Modeling & Explainability**

- Baseline ML + uplift model
- Graph ML (network influence)
- Feature importance / SHAP analysis

**Week 3 – Dashboard & Simulation**

- Build interactive dashboard: scenario selector + simulation
- Visualize top customers, revenue uplift, influence network
- Prepare metrics dashboard (expected revenue, ROI, cost)

**Week 4 – Storytelling & Viral Prep**

- Record 5–10 min demo video (LinkedIn/YouTube friendly)
- Prepare story: problem → AI solution → measurable impact → prototype demo
- Optional blog / LinkedIn article with visuals & screenshots

---

### **8️⃣ Deliverables**

1. **GitHub repo:** structured, Jupyter notebooks, scripts, README
2. **Interactive dashboard:** Streamlit/Gradio
3. **Model artifacts:** baseline + uplift + graph
4. **Scenario simulation report:** Excel/CSV + charts
5. **Storytelling content:** video demo 5–10 menit + LinkedIn post / blog

---

### **9️⃣ Wow Factor**

- **Multi-layer AI:** ML + uplift + Graph ML + generative AI (synthetic data)
- **Interactive & prescriptive:** stakeholder bisa coba skenario sendiri → langsung actionable
- **Measurable impact:** revenue uplift simulation → terlihat nyata, bukan sekadar demo
- **Professional storytelling:** siap viral di LinkedIn / YouTube

---

### **1️⃣0️⃣ Checklist Harian / Milestones**

| Hari | Task                                        |
| ---- | ------------------------------------------- |
| 1–2  | Data prep + synthetic generator             |
| 3–4  | Feature engineering + baseline ML           |
| 5    | Uplift modeling + Graph ML                  |
| 6    | SHAP explainability + network visualization |
| 7–8  | Dashboard setup + interactive scenarios     |
| 9    | Scenario simulation + validate output       |
| 10   | Storyboarding demo video + screenshot       |
| 11   | Record demo + LinkedIn/YouTube upload       |
| 12   | Blog/article writeup + code cleanup         |

---

Beb, ini roadmap + konsep project **dirancang untuk viral, profesional, measurable, dan lebih advanced dari Rio**, karena:

- Ada **multi-layer ML + Graph + Generative AI**
- Ada **interactive simulation dashboard**
- Ada **storytelling dengan measurable business impact**

---

### 1. **Codingnya harus di laptop atau bisa Google Colab?**

- **Laptop lokal**: lu install Python, Jupyter, environment, library dsb → ribet, makan space, kadang error install package.
- **Google Colab**: udah siap, tinggal import library → ga perlu install ribet, gratis GPU, gampang share link.
  👉 Jadi kalau buat projek showcase/viral, **Colab lebih cocok**, tinggal bikin notebook interaktif + bisa dipublish.

---

### 2. **Hasil akhirnya kayak apa sih projek ini?**

Ga cuma tabel atau angka doang, tapi lu bisa bikin **cerita interaktif**. Contoh:

- Ada **dashboard/visualisasi** → grafik customer segmentasi, heatmap, funnel analysis.
- Ada **simulation** → misal: _“Kalau gw naikin promo ke segmen Gen Z, revenue naik 12%.”_
- Ada **AI interaktif** → user bisa input data (usia, paket internet, lokasi), terus AI prediksi:
  _“Customer ini 75% bakal churn dalam 1 bulan kalau ga dikasih promo.”_

👉 Jadi hasil akhirnya bukan cuma kalimat doang, tapi kombinasi:

- **Tabel** → hasil segmentasi pelanggan.
- **Grafik/Visual** → buat storytelling.
- **Kalimat Prediksi** → contoh output AI.
- **Simulation Button/Slider** → user bisa cobain _“what if scenario”_.

---

### 3. **Contoh bentuk output real case**

Misal projek lu "Customer Churn Prediction & Simulation":

🔹 **Dashboard utama**:

- Pie chart: 25% pelanggan rawan churn.
- Map: hotspot churn ada di Jawa Barat & Sumatera.
- KPI: ARPU (average revenue per user) drop 15% di segmen usia 18–25.

🔹 **AI Output (kalimat)**:
_"Pelanggan dengan usia 20, paket 20GB, lokasi Bandung → 87% kemungkinan churn dalam 30 hari ke depan."_

🔹 **Simulation (interactive)**:

- Slider: "Diskon promo (%)"
- Output: "Jika kasih diskon 10% → churn turun dari 25% ke 12%."

---

### 🔹 Tampilan Dashboard / Output

#### 1. Ringkasan

```
📊 Total Customers: 12.430
⚠️ Predicted Churn Risk: 27%
💸 Potential Revenue Loss: Rp 1,2 M / bulan
```

---

#### 2. Grafik Segmentasi

_(pie chart misalnya)_

- 27% Rawan Churn
- 55% Stabil
- 18% Loyal

---

#### 3. AI Prediksi Individu (kalimat output)

```
Customer ID: 10432
Age: 22
Location: Bandung
Package: 20GB / bulan
Prediction: 87% kemungkinan churn dalam 30 hari
Rekomendasi: Kasih promo khusus Gen Z → 3GB bonus weekend
```

---

#### 4. Simulation / What-if Analysis

_(ada slider / tombol interaktif)_

**User input:** Diskon 10% untuk segmen usia 18–25
**AI Output:**

```
✅ Prediksi churn turun dari 27% → 15%
✅ Revenue loss turun dari Rp 1,2 M → Rp 650 Juta
```

Kalau naikin promo jadi 20%:

```
✅ Churn turun jadi 8%
⚠️ Tapi margin profit berkurang 12%
```

---

👉 Jadi hasil akhirnya **bukan cuma tabel**, tapi kombinasi:

- **Tabel ringkasan**
- **Grafik visual**
- **Kalimat rekomendasi**
- **Simulasi interaktif** (biar ada wow factor).

---

## 📂 Struktur Dataset (Multi-file ala Telco Marketing AI)

### 1️⃣ `customer_profile.csv`

**Fungsi:** Identitas dasar pelanggan (buat segmentation). 30k pelanggan.
**Kolom:**

- `customer_id`
- `age`
- `gender`
- `city`
- `tenure_months` (lama jadi pelanggan)
- `plan_type` (prepaid, postpaid, combo, data-only)
- `ARPU` (average revenue per user, proxy SES)
- `device_type` (smartphone low-end, mid, flagship)
- `primary_site_id` (buat link ke network KPI)

---

### 2️⃣ `usage_metrics.csv`

**Fungsi:** Liat pola pemakaian → prediksi churn atau revenue uplift. 90k baris (3 bulan × 30k).
**Kolom:**

- `customer_id`
- `date`
- `data_usage_gb` (per bulan)
- `voice_minutes`
- `sms_count`
- `app_usage_social` (jam/bulan, proxy dari traffic category)
- `app_usage_gaming`

---

### 3️⃣ `campaign_history.csv`

**Fungsi:** Dasar untuk **uplift modeling** & A/B testing (kayak kerjaan Rio). ±34.459 baris.
**Kolom:**

- `customer_id`
- `campaign_id`
- `campaign_type` (discount, free quota, cashback, combo offer)
- `offer_date`
- `treatment_group` (1=ditawarin, 0=control)
- `conversion` (1=beli, 0=tidak)
- `uplift_revenue` (delta ARPU setelah campaign)

---

### 4️⃣ `complaints.csv` _(optional, biar lebih real & advanced)_

**Fungsi:** Hubungkan kualitas jaringan & kepuasan pelanggan ke churn. ±3.2k baris (sparse).
**Kolom:**

- `customer_id`
- `complaint_date`
- `complaint_type` (coverage, internet speed, billing, app issue)
- `resolution_time_hours`
- `resolved` (1=ya, 0=belum)

---

### 5️⃣ `network_kpi.csv` _(optional advanced → nambah nilai lu, karena Rio ga main di sini)_

**Fungsi:** Masukin flavor “network analytics” biar project lu beda. 1.200 site.
**Kolom:**

- `site_id`
- `date`
- `availability`
- `trans_packet_loss`
- `trans_tnl`
- `ul_interference`
- `vswr`
- `rsrp`
- `rsrq`
- `prb_usage`
- `max_user`
- `active_user_max`
- `cssr`

---

## ✅ Apakah ini sudah mirip job Rio?

Banget. Bahkan **lebih luas**:

- Rio: fokus ke **uplift modeling, churn, segmentation, A/B test** → ada semua di dataset ini.
- Lu: tambah layer **network KPI** → diajarin dari background lu di telco infra → bikin project lu unik & lebih “bridging marketing ↔ network”.

---

👉 Jadi hasil akhir project lu bisa:

- **Churn prediction** → siapa yang akan cabut.
- **Uplift modeling** → siapa yang bakal beli kalau dikasih promo.
- **Segmentation** → micro-segment by usage, SES, network quality.
- **Root cause insight** → apakah churn karena kompetitor/promo atau karena **jaringan jelek** (yang Rio belum cover).
