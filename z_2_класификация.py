#классификация
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

text = [
    "Ж",#not dead
    "М",#not dead
    "B",#not dead
    "С",#dead
    "A",#not dead
    "10",# dead
    "20",#not dead
    "50",#not dead
    "70",#dead
    "90",#dead
]

lable = [0,0,0,1,0,1,0,0,1,1] #1 - dead, 0 - not dead

text_train, text_test, y_train, y_test = train_test_split(text, lable, test_size = 0.33, random_state=42)

pipe = make_pipeline(
    CountVectorizer(),# задача сделть мешок слов, это подсчет колличества вхождения слов игнорируя их порядковый номер, на выходе получается матрица
    MultinomialNB() #учит верооятности слов 
)

pipe.fit(text_train, y_train)
y_pred = pipe.predict(text_test)


print(f"Accuracy:{accuracy_score(y_test,y_pred)}")

if (pipe.predict(["Ж 90 С"])[0]) == 0:
    print("this person not dead")
else:
    print("this person dead")

if (pipe.predict(["М 10 C"])[0]) == 0:
    print("this person not dead")
else:
    print("this person dead")
