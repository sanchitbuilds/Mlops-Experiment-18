import pandas as pd
df=pd.read_csv("Student_Performance_DT - Student_Performance_DT.csv")
print(df.head())
print(df.isna().sum())
df.shape
df['Internet_Access']=df['Internet_Access'].map({"Yes":1, "No":0})
df['Extracurricular']=df['Extracurricular'].map({"Yes":1, "No":0})
df['Final_Result']=df['Final_Result'].map({"Pass":1, "Fail":0})
X=df[['Study_Hours', 'Attendance', 'Previous_Score', 'Assignments_Completed', 'Sleep_Hours', 'Participation', 'Internet_Access', 'Extracurricular']]
y=df[['Final_Result']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
trad_model=DecisionTreeClassifier(random_state=42)
trad_model.fit(X_train, y_train)
y_pred_trad=trad_model.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix
trad_acc=accuracy_score(y_pred_trad, y_test)
print(trad_acc)
param_grid={
    'criterion':['gini', 'entropy'],
    'max_depth':[3, 5, 7, None],
    'min_samples_split':[2, 5, 10],
    'min_samples_leaf':[1,3,5]
}
from sklearn.model_selection import GridSearchCV
grid_search=GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)
print(grid_search.best_params_)
print(grid_search.best_score_)
tuned_model=grid_search.best_estimator_
print(tuned_model)
y_pred_tuned=tuned_model.predict(X_test)
tuned_acc=accuracy_score(y_pred_tuned, y_test)
print(tuned_acc)
from sklearn.metrics import classification_report
print(classification_report(y_pred_tuned, y_test))
print(classification_report(y_pred_trad, y_test))