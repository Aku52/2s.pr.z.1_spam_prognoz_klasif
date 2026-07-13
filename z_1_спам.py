#спам
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

text = [
    "купи ффон за 100 зубрей",#spam
    "стрелка в офисе в 10:ОО",#not spam
    "вы выграли милллллион, пришлити CVV",#spam
    "отчет продажи за квартал 2025",#not spam
    "срочно позвони сейчас и получи приз",#spam
    "напоминаем про оплату счета"#not spam
]

lable = [1,0,1,0,1,0] #1 - spam, 0 - not spam

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size = 0.33, random_state=42)
print(text_test)
print(y_test)
pipe = make_pipeline(
    CountVectorizer(),# задача сделть мешок слов, это подсчет колличества вхождения слов игнорируя их порядковый номер, на выходе получается матрица
    MultinomialNB() #учит верооятности слов 
)

pipe.fit(text_train, y_train)
y_pred = pipe.predict(text_test)

print(y_pred)
print(f"Accuracy:{accuracy_score(y_test,y_pred)}")
print("New:", pipe.predict(["поздравляем вы победитель пришлите нам свой CVV код"])[0])
print("New:", pipe.predict(["Дообрый вечер встреча завтра в 12:00"])[0])

