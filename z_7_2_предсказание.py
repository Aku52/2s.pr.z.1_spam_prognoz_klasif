# Напишите программу, которая предсказывает цену мороженого на основе температуры воздуха.
# Используйте данные о ценах и температурах по месяцам. 
# Реализуйте два метода прогнозирования: скользящее среднее и линейную регрессию.
# Пользователь вводит номер месяца, программа выводит прогнозируемую цену.

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

price = np.array([1, 1.1, 1.4, 1.7, 2, 2.5, 3, 3.1, 2.3, 2.1, 1.4, 1])
temp = np.array([-5, -2, 5, 7, 15, 23, 30, 32, 26, 16, 10, -5])

print("ВВЕДИТЕ НОМЕР МЕСЯЦА:")
data = input()


def moving_average(series, data):
    
    if len(series) < int(data):
           return np.mean(series)
    return np.mean(series[-(int(data))])

def moving_average_month(series, data):
    
    if len(series) < int(data):
           return np.mean(series)
    return np.mean(series[-(int(data))])

ma_pred = moving_average(price, int(data))
ma_pred = moving_average_month(temp, int(data))


x = np.arange(len(price)).reshape(-1,1)#weeks(0,1,2,3)
y = price

x_1= np.arange(len(temp)).reshape(-1,1)#weeks(0,1,2,3)
y_1 = temp

model = LinearRegression()
model.fit(x,y)

model_month = LinearRegression()
model_month.fit(x_1,y_1)

next_price = np.array([[len(price)]])
linear_pred = model.predict(next_price)[0]

next_month = np.array([[len(temp)]])
linear_pred_month = model_month.predict(next_month)[0]

print(f"with temp:{linear_pred_month} price for icecream in {data}: {linear_pred}")