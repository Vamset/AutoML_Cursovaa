!pip install fedot scikit-learn

pip install fedot[extra]

!pip install --upgrade pip
!pip install numpy
!pip install --upgrade numpy
!pip install fedot --upgrade

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
import pandas as pd
from google.colab import drive

from google.colab import drive
drive.mount('/content/drive')
file_path = '/content/drive/MyDrive/train.csv'

data = pd.read_csv(file_path, delimiter=',', parse_dates=['date'])
data_trimmed = data.tail(50_000)

temp_file_path = '/content/drive/MyDrive/train_trimmed.csv'
data_trimmed.to_csv(temp_file_path, index=False)

task = Task(TaskTypesEnum.ts_forecasting, TsForecastingParams(forecast_length=10))

train_input = InputData.from_csv_time_series(task=task,
                                             file_path=temp_file_path,
                                             delimiter=',',
                                             target_column='sales')

train_data, test_data = train_test_data_setup(train_input)

model = Fedot(problem='ts_forecasting', task_params=task.task_params, timeout=60,  n_jobs=-1, seed=42)

pipeline = model.fit(train_data)

pipeline.show()

forecast = model.forecast(test_data)
print("Forecast:", forecast)

print("Metrics:", model.get_metrics(metric_names=['rmse', 'mae', 'mape'], target=test_data.target))

model.plot_prediction()
