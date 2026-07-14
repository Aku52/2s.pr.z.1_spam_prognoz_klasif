# Напишите программу, которая предсказывает цену мороженого на основе температуры воздуха.
# Используйте данные о ценах и температурах по месяцам. 
# Реализуйте два метода прогнозирования: скользящее среднее и линейную регрессию.
# Пользователь вводит номер месяца, программа выводит прогнозируемую цену.

# Подключение библиотек
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Данные: цена мороженого и температура по месяцам (январь-декабрь)
price = np.array([1, 1.1, 1.4, 1.7, 2, 2.5, 3, 3.1, 2.3, 2.1, 1.4, 1])
temp = np.array([-5, -2, 5, 7, 15, 23, 30, 32, 26, 16, 10, -5])

# Ввод номера месяца от пользователя
print("ВВЕДИТЕ НОМЕР МЕСЯЦА:")
data = input()

# Функция скользящего среднего для цены
def moving_average(series, data):
    if len(series) < int(data):  # если данных меньше запрошенного номера
        return np.mean(series)   # возвращаем среднее по всем
    return np.mean(series[-int(data):]) # среднее последних N значений


# Функция скользящего среднего для температуры (дублирует первую)
def moving_average_month(series, data):
    if len(series) < int(data):
        return np.mean(series)
    return np.mean(series[-int(data):])  # среднее последних N значений


# Вызов функций (переменная перезаписывается)
ma_pred = moving_average(price, int(data))        # прогноз цены (перезаписывается)
ma_pred = moving_average_month(temp, int(data))   # прогноз температуры

# ПОДГОТОВКА ДАННЫХ ДЛЯ ЛИНЕЙНОЙ РЕГРЕССИИ (цена)
x = np.arange(len(price)).reshape(-1, 1)  # номера месяцев (0,1,2,...)
y = price                                 # цена

# ПОДГОТОВКА ДАННЫХ ДЛЯ ЛИНЕЙНОЙ РЕГРЕССИИ (температура)
x_1 = np.arange(len(temp)).reshape(-1, 1)  # номера месяцев (0,1,2,...)
y_1 = temp                                 # температура

# Обучение модели для цены
model = LinearRegression()
model.fit(x, y)

# Обучение модели для температуры
model_month = LinearRegression()
model_month.fit(x_1, y_1)

# Прогноз цены на введённый месяц
next_price = np.array([[len(price)]])        # следующий месяц (12)
linear_pred = model.predict(next_price)[0]   # предсказанная цена

# Прогноз температуры на введённый месяц
next_month = np.array([[len(temp)]])         # следующий месяц (12)
linear_pred_month = model_month.predict(next_month)[0]  # предсказанная температура

# Вывод результата
print(f"with temp:{linear_pred_month} price for icecream in {data}: {linear_pred}")