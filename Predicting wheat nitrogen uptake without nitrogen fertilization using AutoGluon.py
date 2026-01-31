import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from autogluon.tabular import TabularPredictor
from autogluon.tabular import TabularPredictor
# 1.Read and preprocess the data
df = pd.read_csv('E:/wheatN/outliersN0.csv')

# 2.Type conversion: AutoGluon can automatically handle string categories
for col in ['Zone', 'MainClimates', 'Precipitation', 'Temperature', 'Tillage', 'Nmethod', 'NitrogenForm']:
    df[col] = df[col].astype(str)

for col in ['StrawReturn', 'OrganicInput', 'Irrigation']:
    df[col] = df[col].map({'No': 0, 'Yes': 1})

# 3.Splitting into training/test sets
# Keep all features and target columns in the same DataFrame for AutoGluon
# Remove unnecessary input columns: Nmethod, NitrogenForm, N, P, K, Ntimes, Yield, No
drop_cols = ['Nmethod','NitrogenForm','N','P','K','Ntimes','Yield','No']
df_train, df_test = train_test_split(df.drop(columns=drop_cols), 
                                     test_size=0.3, random_state=42)

# 4.AutoGluon model training path
save_path = 'E:/wheatN/0Nuptake'

# 5.Initialize and train the AutoGluon model (preserving the complete structure)
predictor = TabularPredictor(
    label='Nuptake',
    path=save_path,
    problem_type='regression',
    eval_metric='root_mean_squared_error'
).fit(
    train_data=df_train,
    presets='best_quality',
    refit_full=True,
    verbosity=2
)

# 6.Models list
model_names = predictor.model_names()
print("\n📌 All models：", model_names)

# 7.Test set metrics（RMSE / R²）
y_true = df_test['Yield'].values
test_metrics = []
for m in model_names:
    y_pred = predictor.predict(df_test, model=m)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    test_metrics.append({'model': m, 'RMSE_test': round(rmse, 2), 'R2_test': round(r2, 3)})
df_test_metrics = pd.DataFrame(test_metrics)

# 8.Leaderboard（Cross-validation scores, etc.）
lb = predictor.leaderboard(silent=True) 
score_cols = [c for c in lb.columns if 'score' in c]
cv_metrics = lb[['model'] + score_cols + ['stack_level']].copy()

# 9.Merge two tables
full_metrics = pd.merge(df_test_metrics, cv_metrics, on='model')
full_metrics = full_metrics.sort_values(by='RMSE_test')

# 10.Output and save
print("\n【Success】：")
print(full_metrics)

outpath = 'E:/wheatN/0Yield_model_performance.csv'
full_metrics.to_csv(outpath, index=False)
