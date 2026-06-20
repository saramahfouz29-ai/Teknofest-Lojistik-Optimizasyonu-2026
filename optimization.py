import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import sys
import math

print("="*60)
print(" TEKNOFEST Fleet Optimization Engine (Phase 1) - UPDATED RULES")
print("="*60)

# --- 1. Load All Required Files ---
print("\n[1/4] Loading Excel files...")
try:
    df_pred = pd.read_excel("Tahmin_Ciktisi.xlsx")
    df_coords = pd.read_excel("Koordinatlar.xlsx")
    df_rentals = pd.read_excel("Kiralik_Araclar.xlsx")
    df_costs = pd.read_excel("Arac_Kapasite_Maliyet.xlsx")
except FileNotFoundError as e:
    print(f"\n[ERROR] Missing file: {e}")
    print("Please make sure all 4 Excel files are in this folder.")
    sys.exit(1)

# --- 2. Calculate Distances (Haversine Formula) ---
print("[2/4] Calculating true route distances (Rule 6)...")
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in kilometers
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

coords_dict = {}
for _, row in df_coords.iterrows():
    coords_dict[row['Transfer Merkezi']] = (row['Enlem'], row['Boylam'])

# --- 3. Extract Truck Specs & Costs ---
print("[3/4] Preparing vehicle constraints and pricing...")
vehicles = df_costs['Araç Adı'].tolist()
capacity = dict(zip(vehicles, df_costs['Kapasite (desi)']))
rent_fixed = dict(zip(vehicles, df_costs['Kiralık Araç Günlük Kira (TL)']))
rent_km = dict(zip(vehicles, df_costs['Kiralık Araç Kilometre Başına Maliyet (TL)']))
spot_fixed = dict(zip(vehicles, df_costs['Spot Araç Sabit Günlük Maliyet (TL)']))
spot_km = dict(zip(vehicles, df_costs['Spot Kilometre Başına Maliyet (TL)']))

# --- 4. Run Google OR-Tools MILP Optimizer ---
print("[4/4] Running MILP Optimization for all routes...")
results = []
total_network_cost = 0.0

for index, row in df_pred.iterrows():
    origin = row['Çıkış Transfer Merkezi']
    dest = row['Varış Transfer Merkezi']
    date = row['Tarih']
    demand = row['Tahmini_Desi']
    
    distance = 0.0
    if origin in coords_dict and dest in coords_dict:
        lat1, lon1 = coords_dict[origin]
        lat2, lon2 = coords_dict[dest]
        distance = haversine(lat1, lon1, lat2, lon2)
        
    rental_limits = {'Tır': 0, 'Kamyon': 0, 'Hafif Kamyon': 0}
    
    route_rentals = df_rentals[(df_rentals['Çıkış Transfer Merkezi'] == origin) & 
                               (df_rentals['Varış Transfer Merkezi'] == dest)]
                               
    for _, k_row in route_rentals.iterrows():
        vehicle_type = k_row['Araç Türü']
        if vehicle_type in rental_limits:
            rental_limits[vehicle_type] = k_row['Araç sayısı']

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        continue

    # Setup tracking variables
    k_count = {}
    k_desi = {}
    s_count = {}
    s_desi = {}

    for v in vehicles:
        # KURAL 1 & 3: Kiralık araçlar ZORUNLU yola çıkar. Sayıları sabittir.
        limit = int(rental_limits.get(v, 0))
        k_count[v] = limit
        # Bu kiralık araçlara atanacak toplam desi (Sürekli Değişken)
        k_desi[v] = solver.NumVar(0, limit * capacity[v], f'k_desi_{v}')
        
        # Spot araçlar (Tamsayı Değişken)
        s_count[v] = solver.IntVar(0, 100, f's_count_{v}')
        # Bu spot araçlara atanacak toplam desi (Sürekli Değişken)
        s_desi[v] = solver.NumVar(0, solver.infinity(), f's_desi_{v}')
        
        # Spot Kapasite Kısıtı
        solver.Add(s_desi[v] <= s_count[v] * capacity[v])
        
        # YENİ KURAL 1: Spot araç seçilirse en az %10'u dolmak zorundadır!
        solver.Add(s_desi[v] >= s_count[v] * (capacity[v] * 0.10))

    # Talep Karşılama Kısıtı
    solver.Add(sum(k_desi[v] + s_desi[v] for v in vehicles) >= demand)

    # Amaç Fonksiyonu (Maliyet Minimizasyonu)
    cost_expr = []
    for v in vehicles:
        k_cost = rent_fixed[v] + (rent_km[v] * distance)
        s_cost = spot_fixed[v] + (spot_km[v] * distance)
        cost_expr.append(k_count[v] * k_cost) # Kiralık maliyeti her halükarda ödenir
        cost_expr.append(s_count[v] * s_cost)
        
    solver.Minimize(sum(cost_expr))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        route_cost = solver.Objective().Value()
        total_network_cost += route_cost
        
        # Çıktıları Portalın İstediği Formatta Formatla
        for v in vehicles:
            # Eğer o hatta kiralık araç varsa (boş olsa bile eklenir)
            if k_count[v] > 0:
                results.append({
                    'Tarih': date,
                    'Araç Tipi': f"Kiralık {v}",
                    'Çıkış TM': origin,
                    'Varış TM': dest,
                    'Atanan Desi': round(k_desi[v].solution_value(), 2),
                    'Maliyet': round(k_count[v] * (rent_fixed[v] + (rent_km[v] * distance)), 2)
                })
            
            # Eğer spot araç kullanılmışsa
            s_val = int(s_count[v].solution_value())
            if s_val > 0:
                results.append({
                    'Tarih': date,
                    'Araç Tipi': f"Spot {v}",
                    'Çıkış TM': origin,
                    'Varış TM': dest,
                    'Atanan Desi': round(s_desi[v].solution_value(), 2),
                    'Maliyet': round(s_val * (spot_fixed[v] + (spot_km[v] * distance)), 2)
                })

# --- 5. Export Final Results ---
print("\n" + "="*60)
df_results = pd.DataFrame(results)
df_results.to_excel("Optimizasyon_Ciktisi.xlsx", index=False)

print(" SUCCESS! File 'Optimizasyon_Ciktisi.xlsx' has been generated.")
print(f" THE MOST IMPORTANT NUMBER -> TOTAL FLEET COST: {total_network_cost:,.2f} TL")
print("="*60)
