#классификация

# ПОДКЛЮЧЕНИЕ НЕОБХОДИМЫХ БИБЛИОТЕК
from sklearn.model_selection import train_test_split  # для разделения данных на обучающую и тестовую выборки
from sklearn.pipeline import make_pipeline             # для создания пайплайна (последовательности преобразований)
from sklearn.feature_extraction.text import CountVectorizer  # для преобразования текста в "мешок слов" (числовые признаки)
from sklearn.naive_bayes import MultinomialNB          # классификатор Наивный Байес для текстовых данных
from sklearn.metrics import accuracy_score             # для оценки точности модели

# ВХОДНЫЕ ДАННЫЕ: текстовые строки, обозначающие состояние объекта
# Каждая строка - это символ или число, закодированное как текст
text = [
    "Ж",   # not dead (0)
    "М",   # not dead (0)
    "B",   # not dead (0)
    "С",   # dead (1)
    "A",   # not dead (0)
    "10",  # dead (1)
    "20",  # not dead (0)
    "50",  # not dead (0)
    "70",  # dead (1)
    "90",  # dead (1)
]

# МЕТКИ: 1 = dead (мёртв), 0 = not dead (жив)
lable = [0, 0, 0, 1, 0, 1, 0, 0, 1, 1]

# РАЗДЕЛЕНИЕ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ
# test_size=0.33 означает, что 33% данных (3-4 примера) уйдут на тест
# random_state=42 фиксирует случайность для воспроизводимости результата
text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=42)

# ПАЙПЛАЙН: ПОСЛЕДОВАТЕЛЬНОСТЬ ОБРАБОТКИ ДАННЫХ
pipe = make_pipeline(
    CountVectorizer(),# задача сделть мешок слов, это подсчет колличества вхождения слов игнорируя их порядковый номер, на выходе получается матрица
    MultinomialNB() #учит верооятности слов 
)

# ОБУЧЕНИЕ МОДЕЛИ НА ТРЕНИРОВОЧНЫХ ДАННЫХ
pipe.fit(text_train, y_train)  # модель запоминает связь между текстами и метками

# ПРЕДСКАЗАНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ
y_pred = pipe.predict(text_test)  # модель предсказывает метки для тестовых текстов

# ВЫВОД ТОЧНОСТИ МОДЕЛИ
# accuracy_score сравнивает предсказанные метки (y_pred) с правильными (y_test)
# и возвращает долю правильных ответов (от 0 до 1)
print(f"Accuracy:{accuracy_score(y_test,y_pred)}")

# ПРОВЕРКА НА НОВЫХ КОМБИНАЦИЯХ (которых не было в обучении)
# Пример 1: "Ж 90 С" - модель должна определить, dead (1) или not dead (0)
if (pipe.predict(["Ж 90 С"])[0]) == 0:
    print("this person not dead")  # если модель предсказала 0
else:
    print("this person dead")      # если модель предсказала 1

# Пример 2: "М 10 C" - другая комбинация символов
if (pipe.predict(["М 10 C"])[0]) == 0:
    print("this person not dead")
else:
    print("this person dead")