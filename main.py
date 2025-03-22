import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fedot.api.main import Fedot
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Функции для загрузки и обработки данных
def load_data(filepath, date_columns=None):
    return pd.read_csv(filepath, parse_dates=date_columns)

def fill_missing_values(df):

    if 'type' in df.columns:
        df['type'] = df['type'].fillna('Unknown')
    if 'locale' in df.columns:
        df['locale'] = df['locale'].fillna('Unknown')
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('No description')
    if 'transferred' in df.columns:
        df['transferred'] = df['transferred'].fillna(False)
    if 'transactions' in df.columns:
        df['transactions'] = df['transactions'].fillna(0)
    return df

def add_time_features(df, min_date):

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['date'] = (df['date'] - min_date).dt.days  # Преобразование даты в дни
    return df

# Загрузка данных
train = load_data('train.csv', date_columns=['date'])
print(f"Columns in the dataset: {train.columns}")

# Добавление временных признаков
min_date = train['date'].min() 
train = add_time_features(train, min_date)

# Заполнение пропусков
train = fill_missing_values(train)

# Преобразование категориальных признаков в числовые
categorical_columns = ['family']  
train_encoded = pd.get_dummies(train, columns=categorical_columns, drop_first=True)

# Убираем 'sales' из признаков, так как это целевая переменная
important_features = ['onpromotion', 'date', 'month', 'day_of_week', 'is_weekend'] + [col for col in train_encoded.columns if col.startswith('family')]

# Фильтрация данных по выбранным признакам
X = train_encoded[important_features]
y = train_encoded['sales']
max_data_points = 10000
X = X[:min(len(X), max_data_points)]
y = y[:min(len(y), max_data_points)]
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Проверка
print(f"Train data min date: {X_train['date'].min()}")
print(f"Train data max date: {X_train['date'].max()}")
print(f"Test data min date: {X_test['date'].min()}")
print(f"Test data max date: {X_test['date'].max()}")
print(f"Train columns: {X_train.columns}")
print(f"Test columns: {X_test.columns}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Обучение модели
model = Fedot(problem='ts_forecasting', timeout=10, available_operations=['linear', 'ridge', 'lasso'], n_jobs=-1, seed=42)
pipeline = model.fit(features=X_train, target=y_train)

# Прогнозирование на тестовых данных
print(X_test.shape)
predictions = model.predict(X_test)
print(f"y_test shape: {y_test.shape}")
print(f"predictions shape: {predictions.shape}")

# Если количество предсказанных значений и реальных значений не совпадает
if len(predictions) != len(y_test):

    min_len = min(len(y_test), len(predictions))
    y_test = y_test[:min_len]
    predictions = predictions[:min_len]
    print(f"Обрезаны данные до длины {min_len}")

# Оценка качества модели
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# График прогнозов и реальных значений
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_test, label='Исторические данные')
plt.plot(range(len(y_test)), predictions, label='Прогноз')
plt.legend()
plt.show()
