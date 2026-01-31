import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor
import joblib

# === 1.Read the data===
df = pd.read_csv('E:/wheatN/NUE.csv')

# === 2.Categorical variables are mapped to numerical values===
category_maps = {
    'Zone': {'As': 0, 'Af': 1, 'SA': 2, 'Eu': 3, 'BM': 4, 'Oc': 5},
    'MainClimates': {'Equatorial': 0, 'Subarctic': 1, 'Arid': 2, 'Warm': 3},
    'Precipitation': {'Grassland': 0, 'WinterDrought': 1, 'Desert': 2, 'Wetland': 3, 'SummerDrought': 4},
    'Temperature': {'ColdDry': 0, 'No': 1, 'SummerCool': 2, 'SummerWarm': 3, 'SummerHeat': 4, 'Hot Dry': 5},
    'Tillage': {'Tradition': 0, 'Rotary': 1, 'Tillage': 2, 'Notill': 3, 'DeepPine': 4, 'DeepTillage': 5},
    'Nmethod': {'Broadcasting': 0, 'Deep': 1, 'Drip': 2, 'No': 3},
    'StrawReturn': {'No': 0, 'Yes': 1},
    'OrganicInput': {'No': 0, 'Yes': 1},
    'Irrigation': {'No': 0, 'Yes': 1},
    'NitrogenForm': {'Inorganic': 0, 'Organic': 1, 'OrganicInorganic': 2, 'Slow-release': 3, 'Slow-releaseInorganic': 4, 'No': 5}
}
for col, mapping in category_maps.items():
    df[col] = df[col].map(mapping)

# === 3.Features and Objectives===
X = df.drop(columns=['Nuptake', 'NUE', 'No', 'Yield'])
y = df['NUE']

# === 4.Standardized features===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 5.Divide the data into training and testing sets (for final evaluation)===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# === 6.Train the final model===
final_model = TabPFNRegressor()
final_model.fit(X_train, y_train)

# ✅Save the model and the standardizer
joblib.dump(final_model, 'E:/wheatN/tabpfn_model_NUE.pkl')
joblib.dump(scaler, 'E:/wheatN/tabpfn_scaler_NUE.pkl')
print("💾 Success：tabpfn_model_REN.pkl / tabpfn_scaler_REN.pkl")

# === 7.Test set evaluation===
y_pred = final_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("✅RMSE:", rmse)
print("✅R²:", r2)

# === 8.5-fold cross-validation===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []
cv_r2_scores = []

for fold, (train_index, val_index) in enumerate(kf.split(X_scaled)):
    X_train_cv, X_val_cv = X_scaled[train_index], X_scaled[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

    model_cv = TabPFNRegressor()
    model_cv.fit(X_train_cv, y_train_cv)
    y_val_pred = model_cv.predict(X_val_cv)

    rmse_cv = np.sqrt(mean_squared_error(y_val_cv, y_val_pred))
    r2_cv = r2_score(y_val_cv, y_val_pred)

    cv_rmse_scores.append(rmse_cv)
    cv_r2_scores.append(r2_cv)

    print(f"📦 Fold {fold+1}: RMSE={rmse_cv:.4f}, R²={r2_cv:.4f}")

print("\n🎯RMSE:", np.mean(cv_rmse_scores))
print("🎯R²:", np.mean(cv_r2_scores))
 