"""
TEKNOFEST Hepsiburada Lojistik Optimizasyonu - Faz 1 (MVP)
Desi Talep Tahmin Modeli | May 11-17, 2026

Kullanım:
    pip install xgboost pandas openpyxl scikit-learn
    python teknofest_desi_tahmin.py
"""

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ─── XGBRegressor'ı yükle, yoksa fallback ───────────────────────────────────
try:
    from xgboost import XGBRegressor
    MODEL_BACKEND = "XGBoost"
except ImportError:
    print("[UYARI] xgboost bulunamadı. Lütfen: pip install xgboost")
    print("         GradientBoostingRegressor (sklearn) ile devam ediliyor...\n")
    from sklearn.ensemble import GradientBoostingRegressor as _GBR

    class XGBRegressor:
        """XGBRegressor ile aynı arayüzü sağlayan sklearn sarmalayıcı."""
        def __init__(self, n_estimators=500, learning_rate=0.05, max_depth=5,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, **kw):
            self._model = _GBR(
                n_estimators=n_estimators, learning_rate=learning_rate,
                max_depth=max_depth, subsample=subsample,
                random_state=random_state
            )
        def fit(self, X, y):
            self._model.fit(X, y); return self
        def predict(self, X):
            return self._model.predict(X)
        def score(self, X, y):
            return self._model.score(X, y)

    MODEL_BACKEND = "GradientBoosting (sklearn fallback)"

# ─── Sabitler ────────────────────────────────────────────────────────────────
INPUT_FILE   = "Desi_talep.xlsx"
OUTPUT_FILE  = "Tahmin_Ciktisi.xlsx"
FORECAST_START = "2026-05-11"
FORECAST_END   = "2026-05-17"

FEATURE_COLS = [
    "origin_enc", "dest_enc",
    "day_of_week", "day_of_month", "month",
    "week_of_year", "is_weekend",
]

# ─── 1. Veri Yükleme ─────────────────────────────────────────────────────────
print("=" * 60)
print("  TEKNOFEST Desi Talep Tahmin Modeli")
print("=" * 60)
print(f"\n[1/6] Veri yükleniyor: {INPUT_FILE}")

try:
    df = pd.read_excel(INPUT_FILE)
except FileNotFoundError:
    print(f"\n[HATA] '{INPUT_FILE}' bulunamadı.")
    print("       Lütfen script ile aynı klasöre yerleştirin.")
    sys.exit(1)

print(f"      {len(df):,} satır, {df.shape[1]} sütun yüklendi.")
print(f"      Sütunlar: {df.columns.tolist()}")

# ─── 2. Ön İşleme & Özellik Mühendisliği ─────────────────────────────────────
print("\n[2/6] Ön işleme ve özellik mühendisliği...")

df["Tarih"] = pd.to_datetime(df["Tarih"])

route_counts = df.groupby("Tarih").size()
full_days = route_counts[route_counts >= 50].index
df = df[df["Tarih"].isin(full_days)]
print(f"      Dropped {(~df['Tarih'].isin(full_days)).sum()} rows from partial days")
print(f"      Tarih aralığı: {df['Tarih'].min().date()} → {df['Tarih'].max().date()}")
print(f"      Benzersiz tarih sayısı : {df['Tarih'].nunique()}")

# Zaman bazlı özellikler
df["day_of_week"]  = df["Tarih"].dt.dayofweek        # 0=Pzt, 6=Paz
df["day_of_month"] = df["Tarih"].dt.day
df["month"]        = df["Tarih"].dt.month
df["week_of_year"] = df["Tarih"].dt.isocalendar().week.astype(int)
df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

# Kategorik kodlama
le_origin = LabelEncoder()
le_dest   = LabelEncoder()
df["origin_enc"] = le_origin.fit_transform(df["Çıkış Transfer Merkezi"])
df["dest_enc"]   = le_dest.fit_transform(df["Varış Transfer Merkezi"])

n_origins = df["Çıkış Transfer Merkezi"].nunique()
n_dests   = df["Varış Transfer Merkezi"].nunique()
print(f"      Benzersiz çıkış merkezi : {n_origins}")
print(f"      Benzersiz varış merkezi : {n_dests}")

# ─── 3. Model Eğitimi ─────────────────────────────────────────────────────────
print(f"\n[3/6] Model eğitiliyor [{MODEL_BACKEND}]...")

X = df[FEATURE_COLS]
y = df["Toplam Desi"]

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
model.fit(X, y)

train_r2 = model.score(X, y)
print(f"      Eğitim R² skoru: {train_r2:.4f}")
if train_r2 < 0.70:
    print("      [UYARI] R² düşük — veri kalitesini kontrol edin.")

# ─── 4. Tahmin Kapsamını Hazırla ──────────────────────────────────────────────
print(f"\n[4/6] Tahmin kapsamı oluşturuluyor ({FORECAST_START} – {FORECAST_END})...")

forecast_dates = pd.date_range(FORECAST_START, FORECAST_END, freq="D")
unique_routes = (
    df[["Çıkış Transfer Merkezi", "Varış Transfer Merkezi", "origin_enc", "dest_enc"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

n_routes = len(unique_routes)
n_days   = len(forecast_dates)
total    = n_routes * n_days
print(f"      {n_routes} benzersiz rota × {n_days} gün = {total} tahmin")

# ─── 5. Tahmin Vektörü Oluştur ────────────────────────────────────────────────
print("\n[5/6] Tahminler hesaplanıyor...")

rows = []
for _, route in unique_routes.iterrows():
    for d in forecast_dates:
        rows.append({
            "Çıkış Transfer Merkezi": route["Çıkış Transfer Merkezi"],
            "Varış Transfer Merkezi": route["Varış Transfer Merkezi"],
            "origin_enc":  route["origin_enc"],
            "dest_enc":    route["dest_enc"],
            "Tarih":       d,
            "day_of_week":  d.dayofweek,
            "day_of_month": d.day,
            "month":        d.month,
            "week_of_year": d.isocalendar()[1],
            "is_weekend":   int(d.dayofweek >= 5),
        })

forecast_df = pd.DataFrame(rows)
X_forecast  = forecast_df[FEATURE_COLS]
raw_preds   = model.predict(X_forecast)

# Negatif değerleri sıfıra sabitle
clipped = int((raw_preds < 0).sum())
preds   = np.maximum(raw_preds, 0.0)
if clipped:
    print(f"      {clipped} negatif tahmin → 0 olarak düzeltildi.")
else:
    print("      Negatif tahmin yok — tüm değerler geçerli.")

forecast_df["Tahmini_Desi"] = np.round(preds, 2)

# ─── 6. Çıktı Oluştur & Kaydet ────────────────────────────────────────────────
print(f"\n[6/6] Çıktı dosyası hazırlanıyor: {OUTPUT_FILE}")

output = forecast_df[
    ["Çıkış Transfer Merkezi", "Varış Transfer Merkezi", "Tarih", "Tahmini_Desi"]
].copy()
output["Tarih"] = output["Tarih"].dt.strftime("%Y-%m-%d")
output = output.sort_values(
    ["Tarih", "Çıkış Transfer Merkezi", "Varış Transfer Merkezi"]
).reset_index(drop=True)

output.to_excel(OUTPUT_FILE, index=False)
print(f"      {len(output):,} satır '{OUTPUT_FILE}' dosyasına yazıldı.")

# ─── Özet İstatistikler ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ÖZET İSTATİSTİKLER")
print("=" * 60)
print(f"  Toplam tahmin satırı  : {len(output):,}")
print(f"  Tarih aralığı        : {output['Tarih'].min()} → {output['Tarih'].max()}")
print(f"  Tahmini_Desi ort.    : {output['Tahmini_Desi'].mean():,.2f}")
print(f"  Tahmini_Desi min/max : {output['Tahmini_Desi'].min():,.2f} / {output['Tahmini_Desi'].max():,.2f}")

print("\n  İlk 10 satır (örnek):")
print(output.head(10).to_string(index=False))
print("\n  Günlük toplam Desi tahmini:")
daily = output.groupby("Tarih")["Tahmini_Desi"].sum()
for date, total_desi in daily.items():
    print(f"    {date}: {total_desi:>12,.2f} desi")

print("\n" + "=" * 60)
print(f"  Tamamlandı! '{OUTPUT_FILE}' hazır.")
print("=" * 60)
