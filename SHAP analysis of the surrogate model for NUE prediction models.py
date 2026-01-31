# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18

OUT_DIR = r"E:/wheatN" if os.path.exists(r"E:/wheatN") else r"/mnt/data"

def _to1d(a):
    a = np.asarray(a, dtype=float).ravel()
    return a

def metrics(y_true, y_pred):
    y_true = _to1d(y_true)
    y_pred = _to1d(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)  
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))        
    rho = np.corrcoef(y_true, y_pred)[0, 1] if (y_true.std()>0 and y_pred.std()>0) else np.nan
    return r2, rmse, mae, rho

def parity_plot(y_true, y_pred, title, outfile):
    y_true = _to1d(y_true); y_pred = _to1d(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    r2, rmse, mae, rho = metrics(y_true, y_pred)

    lim_min = float(np.nanmin(np.concatenate([y_true, y_pred])))
    lim_max = float(np.nanmax(np.concatenate([y_true, y_pred])))
    pad = 0.02 * (lim_max - lim_min) if lim_max > lim_min else 1.0
    a, b = lim_min - pad, lim_max + pad

    plt.figure(figsize=(6, 6), dpi=200)
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot([a, b], [a, b], 'r--', lw=1.2)
    plt.title(title)
    plt.xlabel("TabPFN Prediction")
    plt.ylabel("Surrogate Prediction (LightGBM)")
    plt.text(0.04, 0.96,
             f"R²={r2:.3f}\nRMSE={rmse:.3f}\nMAE={mae:.3f}\nρ={rho:.3f}",
             transform=plt.gca().transAxes, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9))
    plt.xlim(a, b); plt.ylim(a, b)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, outfile), dpi=300)
    plt.close()

# ===1.Read the data===
df = pd.read_csv('E:/wheatN/NUE.csv')

# ===2.Type conversion===
for col in ['Zone', 'MainClimates', 'Precipitation', 'Temperature', 'Tillage', 'Nmethod', 'NitrogenForm']:
    df[col] = df[col].astype(str)
for col in ['StrawReturn', 'OrganicInput', 'Irrigation']:
    df[col] = df[col].map({'No': 0, 'Yes': 1})

# === 3.Data clarity and feature definition===
categorical_cols = ['Zone', 'MainClimates', 'Precipitation', 'Temperature',
                    'Tillage', 'Nmethod', 'NitrogenForm',
                    'StrawReturn', 'OrganicInput', 'Irrigation']
label = 'NUE'
drop_cols = ['Nuptake', 'Yield', 'site_id']
df_clean = df.drop(columns=drop_cols)

cont_feats = [col for col in df_clean.columns if col not in categorical_cols + [label]]

#Splitting the data into training and test sets
df_train, df_test = train_test_split(df_clean, test_size=0.3, random_state=42)
X_train = df_train.drop(columns=[label]);  y_train = df_train[label]
X_test  = df_test.drop(columns=[label]);   y_test  = df_test[label]

# === 5.Loading the pre-trained TabPFN model and standardizer===
model  = joblib.load("E:/wheatN/tabpfn_model_NUE.pkl")
scaler = joblib.load("E:/wheatN/tabpfn_scaler_NUE.pkl")

# === 6~7.Prepare data for TabPFN===
print("=== TabPFN 特征匹配调试 ===")
numeric_columns_train = X_train.select_dtypes(include=[np.number]).columns.tolist()

success = False
strategy = {"name": None, "cols": None, "enc_map": None}  # `enc_map` is only used for strategy 5.

#Strategy 1: Use only continuous features
if not success and len(cont_feats) == scaler.n_features_in_:
    strategy.update(name="only_cont", cols=cont_feats)
    X_train_tabpfn = X_train[cont_feats].values
    success = True

#Strategy 2: Continuous + Binary
if not success:
    binary_cols = [c for c in ['StrawReturn', 'OrganicInput', 'Irrigation'] if c in X_train.columns]
    if len(cont_feats) + len(binary_cols) == scaler.n_features_in_:
        strategy.update(name="cont+binary", cols=cont_feats + binary_cols)
        X_train_tabpfn = np.hstack([X_train[cont_feats].values, X_train[binary_cols].values])
        success = True

#Strategy 3: All numerical features
if not success and len(numeric_columns_train) == scaler.n_features_in_:
    strategy.update(name="all_numeric", cols=numeric_columns_train)
    X_train_tabpfn = X_train[numeric_columns_train].values
    success = True

#Strategy 4: The first N numerical features
if not success and len(numeric_columns_train) >= scaler.n_features_in_:
    sel = numeric_columns_train[:scaler.n_features_in_]
    strategy.update(name="firstN_numeric", cols=sel)
    X_train_tabpfn = X_train[sel].values
    success = True

#Strategy 5: The first N columns (perform label encoding on non-numeric columns, and save the mapping for reuse on the test set)
if not success and X_train.shape[1] >= scaler.n_features_in_:
    sel = X_train.columns[:scaler.n_features_in_].tolist()
    enc_map = {}
    X_temp = X_train[sel].copy()
    for c in sel:
        if not np.issubdtype(X_temp[c].dtype, np.number):
            vals = X_temp[c].astype(str).unique().tolist()
            enc_map[c] = {v: i for i, v in enumerate(vals)}
            X_temp[c] = X_temp[c].astype(str).map(enc_map[c]).fillna(-1).astype(float)
    strategy.update(name="firstN_labelencode", cols=sel, enc_map=enc_map)
    X_train_tabpfn = X_temp.values
    success = True

if not success:
    raise ValueError(f"The number of features does not match.")

print(f"✔ Usage Strategy: {strategy['name']}")
print(f"✔ Feature column: {strategy['cols']}")

# —— Training set -> Standardization -> TabPFN prediction (as a surrogate target)
X_train_tabpfn_scaled = scaler.transform(X_train_tabpfn)
y_tabpfn_pred_train = model.predict(X_train_tabpfn_scaled)

# —— The test set was constructed using the "same strategy" —— #
def build_tabpfn_X(df_sub, strategy):
    sel = strategy["cols"]
    if strategy["name"] == "firstN_labelencode":
        enc_map = strategy["enc_map"] or {}
        X_tmp = df_sub[sel].copy()
        for c in sel:
            if not np.issubdtype(X_tmp[c].dtype, np.number):
                mp = enc_map.get(c, {})
                X_tmp[c] = X_tmp[c].astype(str).map(mp).fillna(-1).astype(float)
        return X_tmp.values
    else:
        return df_sub[sel].values

X_test_tabpfn = build_tabpfn_X(X_test, strategy)
X_test_tabpfn_scaled = scaler.transform(X_test_tabpfn)
y_tabpfn_pred_test = model.predict(X_test_tabpfn_scaled)

print("✔ TabPFN prediction complete（Train/Test）")

# ===8. Prepare explanatory features for SHAP/surrogate models: continuous standardization + categorical features. OneHot（Train/Test）
scaler_cont = StandardScaler()
X_train_cont_shap = scaler_cont.fit_transform(X_train[cont_feats])
X_test_cont_shap  = scaler_cont.transform(X_test[cont_feats])

ohe_shap = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_train_cat_shap = ohe_shap.fit_transform(X_train[categorical_cols])
X_test_cat_shap  = ohe_shap.transform(X_test[categorical_cols])

X_train_all = np.hstack([X_train_cont_shap, X_train_cat_shap])
X_test_all  = np.hstack([X_test_cont_shap,  X_test_cat_shap])
feature_names = cont_feats + list(ohe_shap.get_feature_names_out(categorical_cols))

# === 9.Training a LightGBM surrogate model (fitting the TabPFN output)
print("Training the LightGBM surrogate model...")
surrogate = LGBMRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
surrogate.fit(X_train_all, y_tabpfn_pred_train)

# === NEW:Surrogate Model Fidelity Metrics (Train/Test)===
yhat_tr = surrogate.predict(X_train_all)
yhat_te = surrogate.predict(X_test_all)

r2_tr, rmse_tr, mae_tr, rho_tr = metrics(y_tabpfn_pred_train, yhat_tr)
r2_te, rmse_te, mae_te, rho_te = metrics(y_tabpfn_pred_test,  yhat_te)

print("\n=== Surrogate → TabPFN fitting metrics ===")
print(f"Train: R²={r2_tr:.4f}, RMSE={rmse_tr:.4f}, MAE={mae_tr:.4f}, ρ={rho_tr:.4f}")
print(f"Test : R²={r2_te:.4f}, RMSE={rmse_te:.4f}, MAE={mae_te:.4f}, ρ={rho_te:.4f}")

# Save
pd.DataFrame([
    {"split":"train","R2":r2_tr,"RMSE":rmse_tr,"MAE":mae_tr,"Pearson_r":rho_tr},
    {"split":"test","R2":r2_te,"RMSE":rmse_te,"MAE":mae_te,"Pearson_r":rho_te},
]).to_csv(os.path.join(OUT_DIR, "tabpfn_surrogate_fidelity.csv"), index=False, encoding="utf-8-sig")

# Diagonal consistency plot
parity_plot(y_tabpfn_pred_train, yhat_tr, "Surrogate vs TabPFN (Train)", "TabPFN_surrogate_fidelity_train.tif")
parity_plot(y_tabpfn_pred_test,  yhat_te, "Surrogate vs TabPFN (Test)",  "TabPFN_surrogate_fidelity_test.tif")

# ===10.SHAP analysis (for the LightGBM model)===
print("SHAP analyzing...")
explainer = shap.TreeExplainer(surrogate)
shap_values = explainer.shap_values(X_train_all)
X_df = pd.DataFrame(X_train_all, columns=feature_names)

feature_name_map = {
    'N': 'N Rate',
    'SoilpH': 'Soil pH',
    'SOC': 'SOC',
    'SoilTN': 'Soil TN',
    'SoilBD': 'Soil BD',
    'Sand': 'Sand %',
    'Silt': 'Silt %',
    'Clay': 'Clay %',
    'P': 'P Rate',
    'K': 'K Rate',
    'Tillage_Tradition': 'Tillage_Tradition',
    'Precipitation_WinterDrought': 'Precipitation_Dry winter',
    'MainClimates_Subarctic': 'MainClimates_Continental',
    'Precipitation_Wetland': 'Precipitation_No dry season',
    'Irrigation_1': 'Irrigation_Yes',
    'StrawReturn_1': 'StrawReturn_Yes',
    'NitrogenForm_Slow-release': 'NForm_Slow-release',
    'Tillage_Rotary': 'Tillage_Rotary',
    'Nmethod_Drip': 'NMethod_Drip',
    'MainClimates_Warm': 'MainClimates_Temperate',
    'Precipitation_Grassland': 'Precipitation_Semi-arid steppe',
    'Temperature_SummerHeat': 'Temperature_Hot summer',
    'NitrogenForm_Slow-releaseInorganic': 'NForm_Slow-releaseInorganic',
    'NitrogenForm_OrganicInorganic': 'NForm_OrganicInorganic'
}
feature_names_ch = [feature_name_map.get(f, f) for f in feature_names]
X_df = pd.DataFrame(X_train_all, columns=feature_names_ch)

# ===11. SHAP summary plot===
print("SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_df, show=False)
plt.xlabel("SHAP Value", fontsize=18, fontname='Arial')
plt.xticks(fontsize=16); plt.yticks(fontsize=16)
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=16)
for l in cbar.get_yticklabels():
    l.set_fontname("Arial")
cbar.set_ylabel("Feature Value", fontsize=16, fontname='Arial', rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "TabPFN_SHAP_dot.tif"), dpi=300, bbox_inches='tight')
plt.show()

# ===12. SHAP bar plot===
print("SHAP bar plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_df, plot_type='bar', show=False)
plt.xticks(fontsize=16)
plt.xlabel("Mean (|SHAP Value|)", fontsize=18, fontname='Arial')
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "TabPFN_SHAP_bar.tif"), dpi=300, bbox_inches='tight')
plt.show()


