# TEKNOFEST 2026 - Hepsiburada Logistics Optimization (Phase 1)

## Overview
This repository contains the logistics optimization solution for the period of May 11-17, 2026.

## Technical Approach
- **Demand Forecasting:** XGBoost Regressor used to predict daily desi demand.
- **Fleet Optimization:** Google OR-Tools (MILP) solver used to minimize transport costs under rental and capacity constraints.

## Final Result
**Total Fleet Cost: 9,948,347.49 TL**

## Included Files
- `teknofest_desi_tahmin.py`: Demand forecasting pipeline.
- `optimization.py`: MILP Fleet Optimization engine.
- `Tahmin_Ciktisi.xlsx`: AI-generated demand forecast.
- `Optimizasyon_Ciktisi.xlsx`: Optimized fleet assignment schedule.
