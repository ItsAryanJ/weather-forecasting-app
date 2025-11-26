# main.py (FastAPI)
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
from pathlib import Path
import itertools
import math
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# Use the dataset path you uploaded
CSV_PATH = "data/india_2000_2024_daily_weather.csv"

app = FastAPI(title="Weather Forecast API (SARIMAX)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def normalize_cities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['city'] = df['city'].astype(str).str.strip().str.title()
    city_corrections = {
        "Bengaluru": "Bangalore",
        "Bangalore Urban": "Bangalore",
        "New Delhi": "Delhi",
        "Madras": "Chennai"
    }
    df['city'] = df['city'].replace(city_corrections)
    return df

def prepare_monthly(df_city: pd.DataFrame) -> pd.DataFrame:
    df_city = df_city.copy()
    df_city['date'] = pd.to_datetime(df_city['date'])
    df_city = df_city.sort_values('date').set_index('date')
    numeric_cols = [
        'temperature_2m_max',
        'temperature_2m_min',
        'precipitation_sum',
        'wind_speed_10m_max',
        'rain_sum'
    ]
    for c in numeric_cols:
        if c in df_city.columns:
            df_city[c] = pd.to_numeric(df_city[c], errors='coerce')

    if 'temperature_2m_max' in df_city.columns and 'temperature_2m_min' in df_city.columns:
        df_city['avg_temp'] = (df_city['temperature_2m_max'] + df_city['temperature_2m_min']) / 2
    else:
        df_city['avg_temp'] = df_city.get('avg_temp', np.nan)

    monthly = df_city.resample('M').mean(numeric_only=True)

    keep_cols = []
    for c in ['temperature_2m_max','temperature_2m_min','avg_temp','precipitation_sum','wind_speed_10m_max','rain_sum']:
        if c in monthly.columns:
            keep_cols.append(c)

    monthly = monthly[keep_cols].copy()
    monthly = monthly.fillna(method='ffill').fillna(method='bfill')
    return monthly

def fit_sarimax_and_forecast(series: pd.Series, exog: Optional[pd.DataFrame], steps=12):
    series = series.replace([np.inf, -np.inf], np.nan)
    if series.isna().all():
        return np.repeat(np.nan, steps)
    series = series.fillna(series.mean())
    if exog is not None:
        exog = exog.replace([np.inf, -np.inf], np.nan).fillna(exog.mean())
        series, exog = series.align(exog, join='inner', axis=0)
    else:
        exog = None

    if len(series) < 24:
        return np.repeat(series.mean(), steps)

    try:
        model = SARIMAX(series,
                        exog=exog,
                        order=(1,1,1),
                        seasonal_order=(1,1,1,12),
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        res = model.fit(disp=False, maxiter=50)
        if exog is None or exog.shape[0] == 0:
            forecast = res.forecast(steps=steps)
        else:
            future_exog = np.tile(exog.mean().values, (steps, 1))
            forecast = res.forecast(steps=steps, exog=future_exog)
        return np.array(forecast)
    except Exception:
        return np.repeat(series.mean(), steps)

# Load CSV
csv_path = Path(CSV_PATH)
if not csv_path.exists():
    raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Place the file accordingly.")

_raw_df = pd.read_csv(csv_path, parse_dates=['date'])
_raw_df = normalize_cities(_raw_df)

# Supported cities & metrics
CITIES = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai"]

METRICS_MAP = {
    'AvgMaxTemp': 'temperature_2m_max',
    'AvgMinTemp': 'temperature_2m_min',
    'AvgTemp': 'avg_temp'
}

FORECAST_CACHE: Dict[str, Any] = {}
ACCURACY_TABLE = []

months_2025 = pd.date_range("2025-01-31", periods=12, freq="M").strftime("%b-%Y").tolist()

# Precompute forecasts & accuracy
for city in CITIES:
    city_df = _raw_df[_raw_df['city'].str.contains(city, case=False)].copy()
    monthly = prepare_monthly(city_df)

    exog_cols = []
    if 'precipitation_sum' in monthly.columns:
        exog_cols.append('precipitation_sum')
    if 'wind_speed_10m_max' in monthly.columns:
        exog_cols.append('wind_speed_10m_max')
    exog = monthly[exog_cols].copy() if exog_cols else None

    for label, col in METRICS_MAP.items():
        if col not in monthly.columns and label != 'AvgTemp':
            continue

        if label in ['AvgMaxTemp', 'AvgMinTemp', 'AvgTemp']:
            if label == 'AvgTemp' and 'avg_temp' not in monthly.columns:
                continue
            series = monthly[METRICS_MAP[label] if METRICS_MAP[label] in monthly.columns else 'avg_temp']
        else:
            if col not in monthly.columns:
                continue
            series = monthly[col]

        if series.dropna().shape[0] < 36:
            forecast_2025 = fit_sarimax_and_forecast(series, exog, steps=12)
            # normalise into list of floats / None
            FORECAST_CACHE[f"{city}::{label}"] = [None if np.isnan(x) else float(x) for x in forecast_2025]
            continue

        series_clean = series.dropna()
        train = series_clean[:-12]
        test = series_clean[-12:]
        if train.shape[0] < 12:
            pred_test = fit_sarimax_and_forecast(series_clean, exog, steps=12)
        else:
            train_exog = exog.loc[train.index] if exog is not None else None
            pred_test = fit_sarimax_and_forecast(train, train_exog, steps=12)

        test_vals = test.values
        pred_vals = np.array(pred_test[:len(test_vals)])
        denom = np.where(test_vals == 0, 1e-6, test_vals)
        mape = (np.mean(np.abs((test_vals - pred_vals) / denom))) * 100
        acc_pct = 100 - mape

        ACCURACY_TABLE.append({
            "city": city,
            "metric": label,
            "mape_pct": round(mape, 2),
            "accuracy_pct": round(acc_pct, 2)
        })
        
        forecast_2025 = fit_sarimax_and_forecast(series, exog, steps=12)
        FORECAST_CACHE[f"{city}::{label}"] = [None if np.isnan(x) else float(x) for x in forecast_2025]

    # ================================================================
    # FIXED RAINFALL MODEL — MATCHES YOUR ML NOTEBOOK EXACTLY
    # ================================================================
    if city.lower() == "delhi":

        # Monthly rainfall TOTAL (sum) — matches your ML code 
        rain_series = city_df.resample("M", on="date")["rain_sum"].sum().fillna(0)

        # Train full data up to 2024 
        train = rain_series[rain_series.index.year <= 2024]

        # ---------- SAME MODEL AS YOUR ML CODE ----------
        series_log = np.log1p(train)

        model = SARIMAX(series_log,
                        order=(2,1,2),
                        seasonal_order=(1,1,1,12),
                        trend='c',
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        fit = model.fit(disp=False)

        # ---------- Forecast 2025 ----------
        f_log = fit.forecast(steps=12)
        pred_2025 = np.expm1(f_log).values.tolist()

        # Save for frontend
        FORECAST_CACHE["Delhi::RainfallForecast2025"] = {
            "months": months_2025,
            "predicted": [float(x) for x in pred_2025],
            "actual": [
                6.5, 3.4, 0.0, 0.7, 95.7, 107.1,
                259.3, 400.1, 136.1, 89.3, None, None
            ]
        }
        # ============================
        # ACCURACY FOR 2025 (MATCH ML)
        # ============================

        # === ACCURACY FOR RAINFALL 2025 (MATCH ML NOTEBOOK EXACTLY) ===

        # ML uses only Jan–Oct actual data
        actual_2025_ml = np.array([
            6.5, 3.4, 0.0, 0.7, 95.7,
            107.1, 259.3, 400.1, 136.1, 89.3
        ], dtype=float)

        # Use only first 10 predicted months
        pred_2025_ml = np.array(pred_2025[:10], dtype=float)

        # avoid division by zero
        denom = actual_2025_ml.copy()
        denom[denom == 0] = 1e-6

        mape_2025_ml = np.mean(np.abs((actual_2025_ml - pred_2025_ml) / denom)) * 100
        acc_2025_ml = 100 - mape_2025_ml

        # add final values
        ACCURACY_TABLE.append({
            "city": "Delhi",
            "metric": "Rainfall2025",
            "mape_pct": round(mape_2025_ml, 2),
            "accuracy_pct": round(acc_2025_ml, 2)
        })




        # ---------- Testing 2024 (optional) ----------
        test2024 = rain_series[rain_series.index.year == 2024]
        if test2024.shape[0] == 12:
            pred_2024_log = fit.forecast(steps=12)
            pred_2024 = np.expm1(pred_2024_log).values.tolist()
        else:
            pred_2024 = []

        FORECAST_CACHE["Delhi::RainfallTest2024"] = {
            "months": test2024.index.strftime("%b-%Y").tolist(),
            "actual": [float(x) for x in test2024.values.tolist()],
            "predicted": [float(x) for x in pred_2024]
        }
        # ============================
        # ACCURACY FOR 2024 (MATCH ML)
        # ============================

        if test2024.shape[0] == 12:
            actual_2024 = test2024.values.astype(float)
            pred_2024_arr = np.array(pred_2024[:12], dtype=float)

            denom = actual_2024.copy()
            denom[denom == 0] = 1e-6

            mape_2024_ml = np.mean(np.abs((actual_2024 - pred_2024_arr) / denom)) * 100
            acc_2024_ml = 100 - mape_2024_ml

            ACCURACY_TABLE.append({
                "city": "Delhi",
                "metric": "Rainfall2024",
                "mape_pct": round(mape_2024_ml, 2),
                "accuracy_pct": round(acc_2024_ml, 2)
            })


# =====================================================================
#   NEW SECTION — INTEGRATE XGBOOST + HYBRID MONTHLY AQI (2024 & 2025)
# =====================================================================

AQI_CSV = "data/city_day.csv"

def safe_mape(true, pred):
    true = np.array(true, dtype=float)
    pred = np.array(pred, dtype=float)
    denom = np.where(true == 0, np.nan, true)
    return np.nanmean(np.abs((true - pred) / denom)) * 100

def load_aqi_data():
    df = pd.read_csv(AQI_CSV)
    df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df = df[df['City'].str.lower() == "delhi"].copy()
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


# =====================================================================
#   1) ULTRA-FINETUNED XGBOOST — DAILY AQI TEST FOR 2024
# =====================================================================

def compute_aqi_2024_test():
    df = load_aqi_data()

    # Build lag & rolling features
    lags = [1,2,3,7,14,30]
    for lag in lags:
        df[f"AQI_lag{lag}"] = df["AQI"].shift(lag)

    rolls = [3,7,14,30]
    for w in rolls:
        df[f"AQI_roll{w}"] = df["AQI"].rolling(w, min_periods=1).mean()
        df[f"AQI_std{w}"] = df["AQI"].rolling(w, min_periods=1).std().fillna(0)

    df = df.dropna(subset=["AQI"]+[f"AQI_lag{i}" for i in lags])

    # Train/test split
    train = df[df['Datetime'] < "2024-01-01"].copy()
    test  = df[(df['Datetime'] >= "2024-01-01") & (df['Datetime'] < "2025-01-01")].copy()

    feature_cols = [c for c in df.columns if c.startswith("AQI_lag") or c.startswith("AQI_roll") or c.startswith("AQI_std")]
    X_train, y_train = train[feature_cols], train["AQI"]
    X_test, y_test   = test[feature_cols], test["AQI"]

    model = XGBRegressor(
        n_estimators=900,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = safe_mape(y_test, preds)
    acc  = 100 - mape

    ACCURACY_TABLE.append({
        "city": "Delhi",
        "metric": "AQI2024XGB",
        "mape_pct": round(mape, 2),
        "accuracy_pct": round(acc, 2)
    })

    FORECAST_CACHE["Delhi::AQI_2024_Test"] = {
        "dates": test["Datetime"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": y_test.tolist(),
        "predicted": preds.tolist()
    }


# =====================================================================
#   2) HYBRID MONTHLY AQI MODEL — PREDICTION FOR 2025
# =====================================================================

def compute_monthly_aqi_2025():
    df = load_aqi_data()
    df["Year"] = df["Datetime"].dt.year
    df["Month"] = df["Datetime"].dt.month

    monthly = df.groupby(["Year","Month"])["AQI"].mean().reset_index()
    monthly = monthly[monthly["Year"] <= 2024]

    # Build full timeline
    start = pd.Timestamp(monthly["Year"].min(), monthly["Month"].min(), 1)
    end = pd.Timestamp(2024, 12, 1)
    timeline = pd.date_range(start, end, freq="MS")

    mf = pd.DataFrame({"YM": timeline})
    mf["Year"]  = mf["YM"].dt.year
    mf["Month"] = mf["YM"].dt.month

    full = mf.merge(monthly, on=["Year","Month"], how="left")
    full["AQI"] = full["AQI"].ffill()

    # Lags
    for lag in range(1,13):
        full[f"lag_{lag}"] = full["AQI"].shift(lag)

    full["roll_3"] = full["AQI"].rolling(3, min_periods=1).mean()
    full["roll_6"] = full["AQI"].rolling(6, min_periods=1).mean()
    full = full.dropna()

    feature_cols = (
        ["Month"] +
        [f"lag_{i}" for i in range(1,13)] +
        ["roll_3","roll_6"]
    )

    X_train = full[feature_cols]
    y_train = full["AQI"]

    # Fit model
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4
    )
    model.fit(X_train, y_train)

    # Predict 2025 months
    forecast_months = pd.date_range("2025-01-01","2025-12-01",freq="MS")
    history = full["AQI"].tolist()

    preds_ml = []
    for d in forecast_months:
        month = d.month
        lag_vals = history[-12:]
        feat = {
            "Month": month,
            **{f"lag_{i+1}": lag_vals[i] for i in range(12)},
            "roll_3": np.mean(history[-3:]),
            "roll_6": np.mean(history[-6:])
        }
        xrow = pd.DataFrame([feat])[feature_cols]
        p = model.predict(xrow)[0]
        preds_ml.append(p)
        history.append(p)

    # seasonal component
    seas = df.groupby("Month")["AQI"].mean()
    hybrid = {i+1: 0.7*seas.iloc[i] + 0.3*preds_ml[i] for i in range(12)}

    # Actual provided
    actual_2025 = {
        1:218,2:214,3:143,4:152,5:138,6:121,7:78,8:89,9:107,10:232
    }
    y_true = [actual_2025.get(i+1, None) for i in range(12)]
    y_pred = [hybrid[i+1] for i in range(12)]

    # compute accuracy on available actuals
    avail_true = [v for v in y_true if v]
    avail_pred = [y_pred[i] for i,v in enumerate(y_true) if v]

    mape = safe_mape(avail_true, avail_pred)
    acc  = 100 - mape

    ACCURACY_TABLE.append({
        "city": "Delhi",
        "metric": "AQI2025Hybrid",
        "mape_pct": round(mape,2),
        "accuracy_pct": round(acc,2)
    })

    FORECAST_CACHE["Delhi::AQI_2025"] = {
        "months": [f"{i:02d}-2025" for i in range(1,13)],
        "predicted": y_pred,
        "actual": y_true
    }


# Run AQI models when backend boots
compute_aqi_2024_test()
compute_monthly_aqi_2025()


# ------------------------------------------------------------
# Build forecast CSV (safe: only include numeric list forecasts)
# ------------------------------------------------------------
forecast_rows = []
for key, fc in FORECAST_CACHE.items():
    # two kinds stored: lists (for normal metrics) or dicts (for rainfall objects)
    if isinstance(fc, list):
        # key like "City::Metric"
        try:
            city, metric = key.split("::")
        except Exception:
            continue
        for i, v in enumerate(fc):
            month_label = months_2025[i] if i < len(months_2025) else f"m{i+1}"
            forecast_rows.append({
                "City": city,
                "Metric": metric,
                "Month": month_label,
                "Predicted_Value": (round(v, 2) if v is not None else None)
            })
    elif isinstance(fc, dict):
        # fc expected to contain "months" and "predicted"
        months_list = fc.get("months", months_2025)
        preds = fc.get("predicted", [])
        try:
            city, metric = key.split("::")
        except Exception:
            # fallback generic naming if key doesn't split
            city = key
            metric = ""
        for i, v in enumerate(preds):
            month_label = months_list[i] if i < len(months_list) else (months_2025[i] if i < len(months_2025) else f"m{i+1}")
            forecast_rows.append({
                "City": city,
                "Metric": metric,
                "Month": month_label,
                "Predicted_Value": (round(v, 2) if v is not None else None)
            })

forecast_df = pd.DataFrame(forecast_rows)
forecast_csv_path = Path("forecasts_2025_from_backend.csv")
try:
    forecast_df.to_csv(forecast_csv_path, index=False)
except Exception as e:
    # if writing fails, keep going but log if needed
    print("ERROR writing forecast CSV:", e)

# Build ACTUALS (same as earlier)
ACTUALS: Dict[str, Dict[str, Any]] = {}

def build_actuals_from_temp_lists(max_list, min_list):
    max_vals = list(max_list) + [None] * (12 - len(max_list))
    min_vals = list(min_list) + [None] * (12 - len(min_list))
    avg_vals = []
    for a, b in zip(max_vals, min_vals):
        if a is None or b is None:
            avg_vals.append(None)
        else:
            avg_vals.append(round((a + b) / 2, 2))
    return {
        "AvgMaxTemp": max_vals,
        "AvgMinTemp": min_vals,
        "AvgTemp": avg_vals,
        "months": months_2025
    }

ACTUALS["Delhi"] = build_actuals_from_temp_lists(
    [20.5, 23.0, 29.0, 36.0, 39.0, 38.0, 33.0, 32.0, 32.0, 31.5],
    [7.6, 10.0, 15.0, 21.0, 25.0, 27.0, 26.0, 25.0, 23.0, 20.1]
)

ACTUALS["Mumbai"] = build_actuals_from_temp_lists(
    [31.2,31.8,33.1,33.5,34.0,32.1,30.1,29.9,30.7,33.6],
    [18.0,19.1,22.0,24.5,27.1,26.2,25.1,24.9,24.5,23.8]
)
ACTUALS["Bangalore"] = build_actuals_from_temp_lists(
    [28.2,30.4,32.8,33.5,32.4,28.8,27.7,27.6,28.1,27.7],
    [15.8,17.3,19.7,21.3,21.2,20.1,19.6,19.6,19.2,19.1]
)
ACTUALS["Hyderabad"] = build_actuals_from_temp_lists(
    [29.1,31.8,35.3,37.4,39.2,32.8,29.7,29.2,29.8,30.1],
    [16.2,18.8,21.9,24.8,26.8,24.2,23.1,22.7,22.6,20.8]
)
ACTUALS["Chennai"] = build_actuals_from_temp_lists(
    [29.2,31.1,33.3,35.2,37.8,36.9,34.8,34.1,33.8,31.7],
    [21.5,22.4,24.2,26.8,28.2,27.3,26.2,25.7,25.3,24.6]
)

# --- Actual Rainfall for Delhi (2024) extracted from dataset ---
# Keep this for compatibility with your frontend actuals endpoint.
ACTUALS["Delhi"]["Rainfall"] = [
    0.22, 1.33, 0.48, 0.09, 0.06, 2.08,
    7.56, 9.32, 11.23, 0.00, 0.00, 1.24
]

# Ensure rainfall placeholders exist
for city in CITIES:
    if city not in ACTUALS:
        ACTUALS[city] = {"months": months_2025}
    ACTUALS[city].setdefault("rain_sum", [None] * 12)
    ACTUALS[city].setdefault("precipitation_sum", [None] * 12)

@app.get("/api/rainfall_2024")
def get_rainfall_2024():
    # Return full structure (months, actual, predicted, mape, accuracy)
    return FORECAST_CACHE.get("Delhi::RainfallTest2024", {})

@app.get("/api/rainfall_2025")
def get_rainfall_2025():
    # Return full structure (months, predicted, actual if available, mape, accuracy)
    return FORECAST_CACHE.get("Delhi::RainfallForecast2025", {})

@app.get("/api/actuals")
def get_actuals():
    return JSONResponse({"actuals": ACTUALS})

class ForecastRequest(BaseModel):
    cities: List[str]
    metrics: Optional[List[str]] = ["AvgTemp"]
    months: Optional[int] = 12

@app.get("/api/cities")
def get_cities():
    available_metrics = list(METRICS_MAP.keys())
    present_cities = sorted(list(set(_raw_df['city'].unique()).intersection(set(CITIES))))
    return {"cities": present_cities, "metrics": available_metrics}

@app.get("/api/accuracy")
def get_accuracy():
    return {"accuracy": sorted(ACCURACY_TABLE, key=lambda x: (x['city'], x['metric']))}

@app.post("/api/forecasts")
def get_forecasts(req: ForecastRequest):
    result = []
    RAIN_METRICS = {"Rainfall"}
    for city in req.cities:
        for metric in req.metrics:
            key = f"{city}::{metric}"
            # enforce rainfall-only-for-delhi rule: if rainfall metric requested for non-Delhi, return error
            if metric == "Rainfall" and city.lower() != "delhi":
                result.append({
                    "city": city,
                    "metric": metric,
                    "months": months_2025[:req.months],
                    "values": None,
                    "error": "Rainfall forecasts are provided for Delhi only."
                })
                continue

            if key not in FORECAST_CACHE:
                possible = [k for k in FORECAST_CACHE.keys() if k.lower().startswith(city.lower()) and k.endswith(metric)]
                if possible:
                    key = possible[0]
            if key in FORECAST_CACHE:
                vals = FORECAST_CACHE[key][:req.months] if isinstance(FORECAST_CACHE[key], list) else FORECAST_CACHE[key].get("predicted", [])[:req.months]
                months = months_2025[:req.months]
                result.append({
                    "city": city,
                    "metric": metric,
                    "months": months,
                    "values": [None if v is None else round(float(v), 2) for v in vals]
                })
            else:
                result.append({
                    "city": city,
                    "metric": metric,
                    "months": months_2025[:req.months],
                    "values": None,
                    "error": "No forecast available (insufficient historical data)"
                })
    return {"forecasts": result}

@app.get("/api/long_term_temp")
def get_long_term_temp():
    """
    Returns backend-prepared Plotly traces + layout for:
    - Historical monthly max/min/avg (2000-2024)
    - Predicted 2025 (Max/Min/Avg)
    - Forecast 2026 (Avg)
    """
    # Prepare Delhi monthly data (use the already loaded _raw_df)
    try:
        df = _raw_df.copy()
    except NameError:
        # fallback: try to read CSV_PATH if _raw_df not present
        df = pd.read_csv(csv_path, parse_dates=['date'])

    delhi_df = df[df['city'].str.contains("Delhi", case=False)].copy()
    delhi_df['date'] = pd.to_datetime(delhi_df['date'])
    # compute avg_temp
    if 'temperature_2m_max' in delhi_df.columns and 'temperature_2m_min' in delhi_df.columns:
        delhi_df['avg_temp'] = (pd.to_numeric(delhi_df['temperature_2m_max'], errors='coerce') +
                                pd.to_numeric(delhi_df['temperature_2m_min'], errors='coerce')) / 2
    else:
        delhi_df['avg_temp'] = pd.to_numeric(delhi_df.get('avg_temp', pd.Series(dtype=float)), errors='coerce')

    # monthly aggregation (use mean for temps as notebook did)
    monthly = delhi_df.resample('M', on='date').mean(numeric_only=True)
    # limit to columns we need and forward fill if needed
    for c in ['temperature_2m_max', 'temperature_2m_min', 'avg_temp']:
        if c not in monthly.columns:
            monthly[c] = np.nan
    monthly = monthly[['temperature_2m_max', 'temperature_2m_min', 'avg_temp']].fillna(method='ffill').fillna(method='bfill')

    # helper to forecast a series using SARIMAX(1,1,1)(1,1,1,12)
    def forecast_series(series, steps=12):
        series = series.dropna()
        if series.shape[0] < 24:
            # fallback: repeat last value if too short
            last = float(series.iloc[-1]) if series.shape[0] > 0 else 0.0
            return np.array([last] * steps)
        try:
            m = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
            res = m.fit(disp=False)
            return res.forecast(steps=steps)
        except Exception:
            # graceful fallback
            meanv = float(series.mean()) if series.shape[0] > 0 else 0.0
            return np.array([meanv] * steps)

    # Predicted 2025 (Max, Min) using monthly series
    max_pred = forecast_series(monthly['temperature_2m_max'], steps=12)
    min_pred = forecast_series(monthly['temperature_2m_min'], steps=12)
    # avg pred as mean of max/min predictions (matches notebook)
    avg_pred = (max_pred + min_pred) / 2.0

    # Forecast 2026: extend avg series by adding predicted 2025 avg then forecast next 12
    extended_avg_series = pd.concat([monthly['avg_temp'], pd.Series(avg_pred, index=pd.date_range("2025-01-31", periods=12, freq="M"))])
    forecast_2026 = forecast_series(extended_avg_series, steps=12)
    dates_2025 = pd.date_range("2025-01-31", periods=12, freq="M")
    dates_2026 = pd.date_range("2026-01-31", periods=12, freq="M")

    # Build Plotly traces (convert numpy -> python lists)
    # historical (2000-2024)
    hist_dates = monthly.index.strftime("%Y-%m-%d").tolist()
    trace_hist_max = {
        "x": hist_dates,
        "y": [None if (math.isnan(x) if isinstance(x, float) else False) else float(x) for x in monthly['temperature_2m_max'].fillna(method='ffill').tolist()],
        "name": "Max Temp",
        "mode": "lines",
        "line": {"color": "red", "width": 2}
    }
    trace_hist_avg = {
        "x": hist_dates,
        "y": [None if (math.isnan(x) if isinstance(x, float) else False) else float(x) for x in monthly['avg_temp'].fillna(method='ffill').tolist()],
        "name": "Avg Temp",
        "mode": "lines",
        "line": {"color": "white", "width": 2}
    }
    trace_hist_min = {
        "x": hist_dates,
        "y": [None if (math.isnan(x) if isinstance(x, float) else False) else float(x) for x in monthly['temperature_2m_min'].fillna(method='ffill').tolist()],
        "name": "Min Temp",
        "mode": "lines",
        "line": {"color": "royalblue", "width": 2}
    }

    # predicted 2025 traces (solid)
    trace_pred2025_max = {
        "x": dates_2025.strftime("%Y-%m-%d").tolist(),
        "y": [float(x) for x in max_pred],
        "name": "Predicted 2025 Max",
        "mode": "lines+markers",
        "line": {"color": "red", "width": 2, "dash": "solid"},
        "marker": {"size": 4}
    }
    trace_pred2025_avg = {
        "x": dates_2025.strftime("%Y-%m-%d").tolist(),
        "y": [float(x) for x in avg_pred],
        "name": "Predicted 2025 Avg",
        "mode": "lines+markers",
        "line": {"color": "white", "width": 2, "dash": "solid"},
        "marker": {"size": 4}
    }
    trace_pred2025_min = {
        "x": dates_2025.strftime("%Y-%m-%d").tolist(),
        "y": [float(x) for x in min_pred],
        "name": "Predicted 2025 Min",
        "mode": "lines+markers",
        "line": {"color": "royalblue", "width": 2, "dash": "solid"},
        "marker": {"size": 4}
    }

    # forecast 2026 avg (dotted)
    trace_forecast2026_avg = {
        "x": dates_2026.strftime("%Y-%m-%d").tolist(),
        "y": [float(x) for x in forecast_2026],
        "name": "Forecast 2026 Avg",
        "mode": "lines+markers",
        "line": {"color": "white", "width": 3, "dash": "dot"},
        "marker": {"size": 5}
    }

    traces = [
        trace_hist_max,
        trace_hist_avg,
        trace_hist_min,
        trace_pred2025_max,
        trace_pred2025_avg,
        trace_pred2025_min,
        trace_forecast2026_avg
    ]

    # Layout (similar to your notebook)
    layout = {
        "title": "🌆 Delhi — 2000–2024 Actual, Predicted 2025 & Forecast 2026 Temperatures",
        "xaxis": {"title": "Years"},
        "yaxis": {"title": "Temperature (°C)", "range": [0, 45]},
        "template": "plotly_dark",
        "hovermode": "x unified",
        "legend": {"title": {"text": "Temperature Type"}},
        "plot_bgcolor": "rgba(10,10,30,1)",
        "paper_bgcolor": "rgba(10,10,30,1)",
        "font": {"color": "white"}
    }

    return {"traces": traces, "layout": layout}

@app.get("/api/forecast_csv")
def download_forecast_csv():
    if not forecast_csv_path.exists():
        raise HTTPException(status_code=404, detail="Forecast CSV missing")
    return FileResponse(str(forecast_csv_path), media_type='text/csv', filename='forecasts_2025_from_backend.csv')


@app.get("/api/aqi_2024_test")
def api_aqi_2024_test():
    return FORECAST_CACHE.get("Delhi::AQI_2024_Test", {})

@app.get("/api/aqi_2025")
def api_aqi_2025():
    return FORECAST_CACHE.get("Delhi::AQI_2025", {})
