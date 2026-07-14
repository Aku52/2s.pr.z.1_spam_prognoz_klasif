#спам
# БИБЛИОТЕКИ ДЛЯ МАШИННОГО ОБУЧЕНИЯ
from sklearn.model_selection import train_test_split  # для разделения данных на обучающую и тестовую выборки
from sklearn.pipeline import make_pipeline             # для создания пайплайна (последовательности преобразований)
from sklearn.feature_extraction.text import CountVectorizer  # для преобразования текста в "мешок слов" (числовые признаки)
from sklearn.naive_bayes import MultinomialNB          # классификатор Наивный Байес для текстовых данных
from sklearn.metrics import accuracy_score             # для оценки точности модели

text = [
    "купи ффон за 100 зубрей",#spam
    "стрелка в офисе в 10:ОО",#not spam
    "вы выграли милллллион, пришлити CVV",#spam
    "отчет продажи за квартал 2025",#not spam
    "срочно позвони сейчас и получи приз",#spam
    "напоминаем про оплату счета"#not spam
]

lable = [1,0,1,0,1,0] #1 - spam, 0 - not spam

# РАЗДЕЛЕНИЕ НА ТРЕНИРОВОЧНУЮ И ТЕСТОВУЮ ВЫБОРКИ
# test_size=0.33 означает, что 33% данных уйдёт на тест (2 примера из 6)
# random_state=42 фиксирует случайность для воспроизводимости результата
text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size=0.33, random_state=42)

# ВЫВОД ТЕСТОВЫХ ДАННЫХ (для проверки, какие примеры попали в тест)
print(text_test)
print(y_test)

# ПАЙПЛАЙН: ПОСЛЕДОВАТЕЛЬНОСТЬ ОБРАБОТКИ
pipe = make_pipeline(
    CountVectorizer(),# задача сделть мешок слов, это подсчет колличества вхождения слов игнорируя их порядковый номер, на выходе получается матрица
    MultinomialNB() #учит верооятности слов 
)

# ОБУЧЕНИЕ МОДЕЛИ
pipe.fit(text_train, y_train)  # передаём тексты и их метки (1 или 0)

# ПРЕДСКАЗАНИЕ НА ТЕСТОВОЙ ВЫБОРКЕ
y_pred = pipe.predict(text_test)  # модель предсказывает метки для тестовых текстов

# ВЫВОД РЕЗУЛЬТАТОВ
print(y_pred) # что предсказала модель
print(f"Accuracy:{accuracy_score(y_test,y_pred)}") # доля правильных ответов
# ПРОВЕРКА НА НОВЫХ ПРИМЕРАХ (которых нет в обучении)
print("New:", pipe.predict(["поздравляем вы победитель пришлите нам свой CVV код"])[0])
print("New:", pipe.predict(["Дообрый вечер встреча завтра в 12:00"])[0])

