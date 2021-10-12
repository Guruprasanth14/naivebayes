
import pandas as pd
import numpy as np



from google.colab import files
uploaded = files.upload()


dataset = pd.read_csv('titanicsurvival.csv')


print(dataset.shape)
print(dataset.head(5))



income_set = set(dataset['Sex'])
dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
print(dataset.head)



X = dataset.drop('Survived',axis='columns')
X

Y = dataset.Survived
Y



X.columns[X.isna().any()]

X.Age = X.Age.fillna(X.Age.mean())



X.columns[X.isna().any()]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25,random_state =0)



from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)


pclassNo = int(input("Enter Person's Pclass number: "))
gender = int(input("Enter Person's Gender 0-female 1-male(0 or 1): "))
age = int(input("Enter Person's Age: "))
fare = float(input("Enter Person's Fare: "))
person = [[pclassNo,gender,age,fare]]
result = model.predict(person)
print(result)

if result == 1:
  print("Person might be Survived")
else:
  print("Person might not be Survived")


y_pred = model.predict(X_test)
print(np.column_stack((y_pred,y_test)))


from sklearn.metrics import accuracy_score
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test, y_pred)*100))
