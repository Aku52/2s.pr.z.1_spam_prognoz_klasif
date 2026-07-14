#прогнозирование

# ПОДКЛЮЧЕНИЕ НЕОБХОДИМЫХ БИБЛИОТЕК
import numpy as np  # для работы с массивами и математическими операциями
from sklearn.linear_model import LinearRegression  # модель линейной регрессии для прогнозирования
from sklearn.metrics import mean_absolute_error  # метрика для оценки точности прогноза (не используется)

# Данные: продажи по неделям
sales = np.array([100, 120, 180, 250, 320, 255, 130, 180, 265, 310])
print(sales)

# Функция скользящего среднего (окно = 3)
def moving_average(series, window=3):
    if len(series) < window:  # если данных меньше окна
        return np.mean(series)  # берём среднее по всем
    return np.mean(series[-window:]) # среднее последних window значений

# Прогноз скользящим средним
ma_pred = moving_average(sales, window=3)
print(f"Moving average prediction to next week:{ma_pred}")

# СОЗДАНИЕ ПРИЗНАКОВ: номера недель (0, 1, 2, 3, ... 9)
# reshape(-1, 1) преобразует массив в столбец (матрицу с 1 столбцом)
# Это нужно, так как sklearn ожидает, что X будет двумерным
x = np.arange(len(sales)).reshape(-1, 1)
print("Номера недель (признак X):\n", x)

# Целевая переменная: продажи
y = sales

# СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ ЛИНЕЙНОЙ РЕГРЕССИИ
model = LinearRegression()  # создаём объект модели
model.fit(x, y)  # обучаем модель на данных (ищем коэффициенты: наклон и пересечение)

# ПРОГНОЗ НА СЛЕДУЮЩУЮ НЕДЕЛЮ
# len(sales) = 10, значит следующая неделя имеет номер 10
next_week = np.array([[len(sales)]])  # создаём массив с номером недели 10
print("Номер следующей недели:", next_week)

# model.predict() возвращает массив с предсказанными значениями
# [0] — берём первое (и единственное) значение прогноза
linear_pred = model.predict(next_week)[0]

print(f"linear regression prediction next week:{linear_pred}")