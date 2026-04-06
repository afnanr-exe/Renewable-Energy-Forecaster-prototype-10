import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Absolute paths
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR  = os.path.join(BASE_DIR, "output", "plots")
MODELS_DIR = os.path.join(BASE_DIR, "output", "models")

# Units
UNITS = {
    "Wind": "MW",
    "Solar": "MW"
}


def train_test_split_by_time(df, target, test_days=183):
    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    train = df[df["timestamp"] <= cutoff]
    test = df[df["timestamp"] > cutoff]
    return train, test


def add_lags(df, target, lags=(1,)):
    df = df.copy()
    for lag in lags:
        df[f"lag{lag}"] = df[target].shift(lag)
    return df


def build_equation(beta, feature_names, target, max_len=80):
    terms = [f"{beta[0]:.3f}"]
    for coef, name in zip(beta[1:], feature_names):
        terms.append(f"{coef:.3f}*{name}")

    eq = f"{target} = " + " + ".join(terms)
    wrapped = "\n".join([eq[i:i+max_len] for i in range(0, len(eq), max_len)])
    return wrapped


def compute_metrics(y_test, y_pred):
    residuals = y_test - y_pred

    r2 = 1 - (residuals**2).sum() / ((y_test - y_test.mean())**2).sum()
    rmse = np.sqrt((residuals**2).mean())
    mae = np.abs(residuals).mean()

    output_range = y_test.max() - y_test.min()
    rmse_pct = (rmse / output_range) * 100
    mae_pct = (mae / output_range) * 100

    return r2, rmse, mae, rmse_pct, mae_pct


def save_plots(prefix, target, timestamps, y_test, y_pred,
               equation, r2, rmse, mae, rmse_pct, mae_pct, label):

    os.makedirs(PLOTS_DIR, exist_ok=True)
    unit = UNITS.get(label, UNITS.get(label.capitalize(), "MW"))

    # ================= SCATTER =================
    plt.figure(figsize=(10, 8))

    plt.scatter(y_test, y_pred, alpha=0.5, label="Predicted")

    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'k--', label="Ideal (y = x)")

    plt.xlabel(f"Actual {target} ({unit})", fontsize=12)
    plt.ylabel(f"Predicted {target} ({unit})", fontsize=12)
    plt.title(f"{label} — Actual vs Predicted (Test Set)", fontsize=14)

    plt.text(0.01, 0.99,
             f"{equation}\n"
             f"Test R² = {r2:.4f}\n"
             f"RMSE = {rmse:.2f} {unit} ({rmse_pct:.2f}%)\n"
             f"MAE = {mae:.2f} {unit} ({mae_pct:.2f}%)",
             transform=plt.gca().transAxes,
             fontsize=9,
             verticalalignment="top")

    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    scatter_path = os.path.join(PLOTS_DIR, f"{prefix.lower()}_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()

    # ================= TIME SERIES =================
    plt.figure(figsize=(14, 6))

    plt.plot(timestamps, y_test, label=f"Actual ({unit})")
    plt.plot(timestamps, y_pred, '--', label=f"Predicted ({unit})")

    plt.title(f"{label} — Test Set Time Series", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(f"{label} Output ({unit})", fontsize=12)

    plt.legend(fontsize=10)
    plt.grid(True)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    ts_path = os.path.join(PLOTS_DIR, f"{prefix.lower()}_timeseries.png")
    plt.savefig(ts_path, dpi=150)
    plt.close()

    return scatter_path, ts_path


def run_both_models(csv_path, target, features, label, test_days=183):

    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=[target] + features)

    if len(df) < 10:
        raise ValueError(f"Not enough data for {label}")

    # ================= LINEAR =================
    df_lin = add_lags(df, target, lags=(1,))
    df_lin = df_lin.dropna()
    features_lin = features + ["lag1"]

    train_lin, test_lin = train_test_split_by_time(df_lin, target, test_days)

    X_train = train_lin[features_lin].to_numpy()
    y_train = train_lin[target].to_numpy()
    X_test = test_lin[features_lin].to_numpy()
    y_test = test_lin[target].to_numpy()

    X_train = np.column_stack([np.ones(len(X_train)), X_train])
    X_test = np.column_stack([np.ones(len(X_test)), X_test])

    beta_lin = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
    y_pred_lin = X_test @ beta_lin

    r2_lin, rmse_lin, mae_lin, rmse_pct_lin, mae_pct_lin = compute_metrics(y_test, y_pred_lin)
    eq_lin = build_equation(beta_lin, features_lin, target)

    scatter_lin, ts_lin = save_plots(
        f"{label}_linear",
        target,
        test_lin["timestamp"],
        y_test,
        y_pred_lin,
        eq_lin,
        r2_lin,
        rmse_lin,
        mae_lin,
        rmse_pct_lin,
        mae_pct_lin,
        label
    )

    # ================= POLYNOMIAL =================
    df_poly = add_lags(df, target, lags=(1, 2))
    df_poly = df_poly.dropna()
    features_poly = features + ["lag1", "lag2"]

    df_ext = df_poly.copy()

    for f in features_poly:
        df_ext[f"{f}_sq"] = df_ext[f] ** 2

    for i in range(len(features_poly)):
        for j in range(i+1, len(features_poly)):
            f1, f2 = features_poly[i], features_poly[j]
            df_ext[f"{f1}_x_{f2}"] = df_ext[f1] * df_ext[f2]

    all_features = df_ext.columns.drop(["timestamp", target])

    train_poly, test_poly = train_test_split_by_time(df_ext, target, test_days)

    X_train_p = train_poly[all_features].to_numpy()
    y_train_p = train_poly[target].to_numpy()
    X_test_p = test_poly[all_features].to_numpy()
    y_test_p = test_poly[target].to_numpy()

    X_train_p = np.column_stack([np.ones(len(X_train_p)), X_train_p])
    X_test_p = np.column_stack([np.ones(len(X_test_p)), X_test_p])

    beta_poly = np.linalg.lstsq(X_train_p, y_train_p, rcond=None)[0]
    y_pred_poly = X_test_p @ beta_poly

    r2_poly, rmse_poly, mae_poly, rmse_pct_poly, mae_pct_poly = compute_metrics(y_test_p, y_pred_poly)
    eq_poly = build_equation(beta_poly, all_features, target)

    scatter_poly, ts_poly = save_plots(
        f"{label}_poly",
        target,
        test_poly["timestamp"],
        y_test_p,
        y_pred_poly,
        eq_poly,
        r2_poly,
        rmse_poly,
        mae_poly,
        rmse_pct_poly,
        mae_pct_poly,
        label
    )

    # ================= BEST MODEL =================
    best = "polynomial" if r2_poly >= r2_lin else "linear"
    best_r2 = r2_poly if r2_poly >= r2_lin else r2_lin
    best_rmse = rmse_poly if r2_poly >= r2_lin else rmse_lin
    best_rmse_pct = rmse_pct_poly if r2_poly >= r2_lin else rmse_pct_lin
    best_mae = mae_poly if r2_poly >= r2_lin else mae_lin
    best_mae_pct = mae_pct_poly if r2_poly >= r2_lin else mae_pct_lin

    return {
        "linear": {
            "r2":   round(r2_lin,   4),
            "rmse": round(rmse_lin, 2),
            "rmse_pct": round(rmse_pct_lin, 2),
            "mae":  round(mae_lin,  2),
            "mae_pct": round(mae_pct_lin, 2),
            "equation": eq_lin,
            "scatter_plot":       scatter_lin,
            "timeseries_plot":    ts_lin,
        },
        "polynomial": {
            "r2":   round(r2_poly,   4),
            "rmse": round(rmse_poly, 2),
            "rmse_pct": round(rmse_pct_poly, 2),
            "mae":  round(mae_poly,  2),
            "mae_pct": round(mae_pct_poly, 2),
            "equation": eq_poly,
            "scatter_plot":       scatter_poly,
            "timeseries_plot":    ts_poly,
        },
        "best_model": best,
        "best_r2": round(best_r2, 4),
        "best_rmse": round(best_rmse, 2),
        "best_rmse_pct": round(best_rmse_pct, 2),
        "best_mae": round(best_mae, 2),
        "best_mae_pct": round(best_mae_pct, 2),
    }
