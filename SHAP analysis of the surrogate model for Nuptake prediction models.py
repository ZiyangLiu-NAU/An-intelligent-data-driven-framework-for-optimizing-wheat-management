# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import shap
import matplotlib.pyplot as plt
import os


plt.rcParams['font.family'] = ['Arial', 'SimSun']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16

OUT_DIR = r"E:/wheatN" if os.path.exists(r"E:/wheatN") else r"/mnt/data"

# 1.Read the data
df = pd.read_csv('E:/wheatN/wheatN.csv')
df = df[df['N'] != 0]

categorical_cols = [
    'Zone','MainClimates','Precipitation','Temperature',
    'Tillage','Nmethod','NitrogenForm','StrawReturn','OrganicInput','Irrigation'
]
cont_feats = ['N','SoilpH','SOC','SoilTN','SoilBD','Sand','Silt','Clay','P','K']
drop_cols = ['Yield', 'No']
label = 'Nuptake'

# 2.Training/testing split (aligned with the feature set used by AutoGluon: drop `drop_cols`, keep the `label` column so that the predictor ignores it)
df_train, df_test = train_test_split(df.drop(columns=drop_cols), test_size=0.3, random_state=42)

# 3.Loading the AutoGluon predictor (ensemble/final model)
predictor = TabularPredictor.load("E:/wheatN/Nuptake", require_version_match=False)

# 4.Using AutoGluon's predictions as the "ground truth" y, train a surrogate model to fit it
X_train_df = df_train.drop(columns=[label])
X_test_df  = df_test.drop(columns=[label])
y_ens_pred_train = predictor.predict(df_train).astype(float).values
y_ens_pred_test  = predictor.predict(df_test).astype(float).values

# 5.Feature preprocessing (transformation is fitted only on the training set)
scaler = StandardScaler()
X_train_cont = scaler.fit_transform(X_train_df[cont_feats])
X_test_cont  = scaler.transform(X_test_df[cont_feats])

ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_train_cat = ohe.fit_transform(X_train_df[categorical_cols])
X_test_cat  = ohe.transform(X_test_df[categorical_cols])

X_train_all = np.hstack([X_train_cont, X_train_cat])
X_test_all  = np.hstack([X_test_cont,  X_test_cat])

feature_names = cont_feats + list(ohe.get_feature_names_out(categorical_cols))

# 6.LightGBM surrogate model
surrogate = LGBMRegressor(random_state=42)
surrogate.fit(X_train_all, y_ens_pred_train)

# ======Fidelity Metrics======
def _to1d(a):
    a = np.asarray(a, dtype=float).ravel()
    return a

def metrics(y_true, y_pred):
    y_true = _to1d(y_true)
    y_pred = _to1d(y_pred)
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    try:
        rmse = mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_t = y_true.std()
    std_p = y_pred.std()
    rho = np.corrcoef(y_true, y_pred)[0, 1] if (std_t > 0 and std_p > 0) else np.nan
    return r2, rmse, mae, rho

yhat_tr = surrogate.predict(X_train_all)
yhat_te = surrogate.predict(X_test_all)

r2_tr, rmse_tr, mae_tr, rho_tr = metrics(y_ens_pred_train, yhat_tr)
r2_te, rmse_te, mae_te, rho_te = metrics(y_ens_pred_test,  yhat_te)

print("=== Surrogate → Ensemble 拟合指标（训练/测试）===")
print(f"Train: R²={r2_tr:.4f}, RMSE={rmse_tr:.4f}, MAE={mae_tr:.4f}, ρ(Pearson)={rho_tr:.4f}")
print(f"Test : R²={r2_te:.4f}, RMSE={rmse_te:.4f}, MAE={mae_te:.4f}, ρ(Pearson)={rho_te:.4f}")

#Save 
pd.DataFrame([{
    "split":"train","R2":r2_tr,"RMSE":rmse_tr,"MAE":mae_tr,"Pearson_r":rho_tr
},{
    "split":"test","R2":r2_te,"RMSE":rmse_te,"MAE":mae_te,"Pearson_r":rho_te
}]).to_csv(os.path.join(OUT_DIR, "ensemble_surrogate_fidelity.csv"), index=False, encoding="utf-8-sig")

#Goodness-of-fit scatter plot with diagonal line
def parity_plot(y_true, y_pred, title, outfile):
    #Ensure it's a one-dimensional floating-point array and remove NaN/Inf values
    def _to1d(a):
        a = np.asarray(a, dtype=float).ravel()
        return a

    y_true = _to1d(y_true)
    y_pred = _to1d(y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]

    #Metrics
    r2, rmse, mae, rho = metrics(y_true, y_pred)

    #Axis range and plotting
    lim_min = float(np.nanmin(np.concatenate([y_true, y_pred])))
    lim_max = float(np.nanmax(np.concatenate([y_true, y_pred])))
    pad = 0.02 * (lim_max - lim_min) if lim_max > lim_min else 1.0
    a, b = lim_min - pad, lim_max + pad

    plt.figure(figsize=(6,6), dpi=200)
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot([a, b], [a, b], 'r--', lw=1.2)
    plt.title(title)
    plt.xlabel("Ensemble Prediction (AutoGluon)")
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


parity_plot(y_ens_pred_train, yhat_tr, "Surrogate vs Ensemble (Train)", "surrogate_fidelity_train.tif")
parity_plot(y_ens_pred_test,  yhat_te, "Surrogate vs Ensemble (Test)",  "surrogate_fidelity_test.tif")

# 7.SHAP analysis (for the LightGBM model)
explainer = shap.TreeExplainer(surrogate)
shap_values = explainer.shap_values(X_train_all)

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
    'Irrigation_Yes': 'Irrigation_Yes',
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

# 8.SHAP summary (dot)
shap.summary_plot(shap_values, X_df, show=False)
plt.xlabel("SHAP Value", fontsize=18, fontname='arial')
plt.xticks(fontsize=16); plt.yticks(fontsize=16)
cbar = plt.gcf().axes[-1]
cbar.tick_params(labelsize=16)
for l in cbar.get_yticklabels():
    l.set_fontname("arial")
cbar.set_ylabel("Feature Value", fontsize=16, fontname='arial', rotation=270, labelpad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap-Nuptake-dot.tif"), dpi=300)
plt.show()

# 9.SHAP summary (bar)
shap.summary_plot(shap_values, X_df, plot_type='bar', show=False)
plt.xlabel("Mean (|SHAP Value|)", fontsize=18)
plt.xticks(fontsize=16); plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap-yield-bar.tif"), dpi=300)
plt.show()

# 10.Output
importance = pd.DataFrame({
    'feature': X_df.columns,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0)
}).sort_values('mean_abs_shap', ascending=False)
importance.to_csv(os.path.join(OUT_DIR, 'ensemble_surrogate_shap_importance_onehot.csv'), index=False, encoding="utf-8-sig")
print(importance.head(10))
