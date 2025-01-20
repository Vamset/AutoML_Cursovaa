import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fedot.api.main import Fedot
import logging
import time
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


logging.basicConfig(
    level=logging.DEBUG,  # Установите уровень отладки
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("debug_logs.log"),  # Сохранение логов в файл
        logging.StreamHandler()  # Отображение логов в консоли
    ]
)

# Функции для загрузки, слияния и обработки данных
def load_data(filepath, date_columns=None):
    return pd.read_csv(filepath, parse_dates=date_columns)


def merge_data(df, merge_df, on_columns, how='left'):
    return df.merge(merge_df, on=on_columns, how=how)


def add_time_features(df, min_date):
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['date'] = (df['date'] - min_date).dt.days
    return df


def fill_missing_values(df):
    df['dcoilwtico'] = df['dcoilwtico'].fillna(method='ffill')
    df['dcoilwtico'] = df['dcoilwtico'].fillna(df['dcoilwtico'].mean())

    df['type'] = df['type'].fillna('Unknown')
    df['locale'] = df['locale'].fillna('Unknown')
    df['locale_name'] = df['locale_name'].fillna('Unknown')
    df['description'] = df['description'].fillna('No description')
    df['transferred'] = df['transferred'].fillna(False)
    df['transactions'] = df['transactions'].fillna(0)
    return df


def create_dummies(df, columns):
    return pd.get_dummies(df, columns=columns, drop_first=True)


# Загрузка данных
train = load_data('train.csv', date_columns=['date'])
test = load_data('test.csv', date_columns=['date'])
stores = load_data('stores.csv')
oil = load_data('oil.csv', date_columns=['date'])
transactions = load_data('transactions.csv', date_columns=['date'])
holidays = load_data('holidays_events.csv', date_columns=['date'])

# Слияние данных
train = merge_data(train, oil[['date', 'dcoilwtico']], on_columns='date')
train = merge_data(train, holidays, on_columns='date')
train = merge_data(train, transactions, on_columns=['date', 'store_nbr'])

test = merge_data(test, oil[['date', 'dcoilwtico']], on_columns='date')
test = merge_data(test, holidays, on_columns='date')
test = merge_data(test, transactions, on_columns=['date', 'store_nbr'])
missing_in_test = set(train.columns) - set(test.columns)

# Добавление временных признаков
min_date = train['date'].min()
train = add_time_features(train, min_date)
test = add_time_features(test, min_date)

# Заполнение пропусков
train = fill_missing_values(train)
test = fill_missing_values(test)


# Создаём список всех категориальных столбцов, которые нужно закодировать
categorical_columns = ['family', 'type', 'locale', 'locale_name', 'description']

# Применяем pd.get_dummies к train и test, добавляя недостающие столбцы в test
train_encoded = pd.get_dummies(train, columns=categorical_columns, drop_first=True)
test_encoded = pd.get_dummies(test, columns=categorical_columns, drop_first=True)

# Обеспечиваем одинаковую структуру train и test
for column in train_encoded.columns:
    if column not in test_encoded.columns:
        test_encoded[column] = 0

for column in test_encoded.columns:
    if column not in train_encoded.columns:
        train_encoded[column] = 0

# Убедимся, что столбцы идут в одинаковом порядке
train_encoded = train_encoded.sort_index(axis=1)
test_encoded = test_encoded.sort_index(axis=1)


#
#
# # Функция для проверки на NaN и бесконечность в данных
# def check_for_nan_and_inf(df, dataset_name="Dataset"):
#     print(f"Проверка для {dataset_name}:")
#
#     # Проверка на NaN
#     if df.isnull().any().any():
#         print("Warning: Data contains NaN values!")
#         print(df.isnull().sum())  # Показываем количество пропусков по каждому столбцу
#     else:
#         print("No NaN values found.")
#
#     # Проверка на бесконечность (исключаем нечисловые столбцы)
#     numeric_df = df.select_dtypes(include=[np.number])  # Оставляем только числовые столбцы
#     if np.isinf(numeric_df.values).any():
#         print("Warning: Data contains infinite values (inf or -inf)!")
#         print((numeric_df == np.inf).sum())  # Показываем количество бесконечных значений по каждому столбцу
#     else:
#         print("No infinite values found.")
#
#     # Статистика по данным
#     print("Общие статистики данных:")
#     print(df.describe())  # Показываем общую статистику по данным
#
#
# # Проверка для исходных данных
# check_for_nan_and_inf(train, "train")
# check_for_nan_and_inf(test, "test")
#
# # Проверка для закодированных данных
# check_for_nan_and_inf(train_encoded, "train_encoded")
# check_for_nan_and_inf(test_encoded, "test_encoded")

important_features = [
    'sales', 'onpromotion', 'date', 'month', 'day_of_week', 'is_weekend',
    'type_Holiday', 'type_Work Day', 'description_Black Friday', 'description_Carnaval',
    'description_Navidad', 'family_BEVERAGES', 'family_BREAD/BAKERY', 'family_DAIRY',
    'family_FROZEN FOODS', 'family_GROCERY I', 'family_GROCERY II', 'family_MEATS',
    'locale_name_Guayaquil', 'locale_name_Quito', 'dcoilwtico'
]

# Фильтруем только нужные столбцы, исключая 'sales' из теста
existing_columns = [col for col in important_features if col in train.columns and col != 'sales']

# Разделяем данные на признаки (X) и целевую переменную (y)
X_train = train[existing_columns]
y_train = train['sales']
X_test = test[existing_columns]

forecast_length = len(test)  # Горизонт прогноза
X_train = train[existing_columns][:10000]
y_train = train['sales'][:10000]
X_test = test[existing_columns]



start_time = time.time()
logging.info("Начало обучения модели")

# model = ARIMA(y_train, order=(1, 1, 1))
#
# # Обучаем модель
# model_fit = model.fit()
#
# # Прогнозируем на будущее
# forecast = model_fit.forecast(steps=10)
#
# # Оценка качества модели
# predictions = model_fit.predict(start=0, end=len(y_train)-1)
# mse = mean_squared_error(y_train, predictions)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_train, predictions)
# print(f'RMSE: {rmse}')
# print(f'MAE: {mae}')
#
# # Показать прогноз и реальное значение
# plt.figure(figsize=(10, 6))
# plt.plot(y_train, label='Исторические данные')
# plt.plot(np.arange(len(y_train), len(y_train)+10), forecast, label='Прогноз')
# plt.legend()
# plt.show()


model = Fedot(problem='ts_forecasting', timeout=60, available_operations=['linear', 'ridge', 'lasso'],n_jobs=-1, seed=42)


pipeline = model.fit(features=X_train, target=y_train)
print("Модель обучена")

predictions = model.predict(features=X_test)

mae = mean_absolute_error(y_train[-len(predictions):], predictions)
rmse = np.sqrt(mean_squared_error(y_train[-len(predictions):], predictions))

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

# График прогнозов и реальных значений
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_train)), y_train, label='Исторические данные')
plt.plot(range(len(y_train), len(y_train) + len(predictions)), predictions, label='Прогноз')
plt.legend()
plt.show()

plot_acf(y_train, lags=50)
plt.title('Autocorrelation of Sales')
plt.show()




submission = pd.DataFrame({'id': test['id'], 'sales': predictions})

submission.to_csv('submission.csv', index=False)

# Визуализация
plt.figure(figsize=(10, 6))
plt.hist(y_train, bins=30, alpha=0.7, color='blue')
plt.title('Распределение продаж')
plt.xlabel('Продажи')
plt.ylabel('Частота')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(oil['date'], oil['dcoilwtico'], label='Цена нефти', color='orange')
plt.title('Цена на нефть во времени')
plt.xlabel('Дата')
plt.ylabel('Цена нефти')
plt.legend()
plt.show()
