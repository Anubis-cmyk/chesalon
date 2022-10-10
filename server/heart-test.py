import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder

#read and train the data set
df = pd.read_csv("heart.csv")

numaricCols = df.select_dtypes([np.number]).columns
dataFrameColums = df.columns
nonNumaricCols = list((Counter(dataFrameColums) - Counter(numaricCols)).elements())

le = LabelEncoder()
for c in nonNumaricCols:
    df[c] = le.fit_transform(df[c])


model = LogisticRegression()
model = LogisticRegression(solver='lbfgs', max_iter=len(df))
columns = list(df.columns)
model.fit(df[columns[:-1]].values,df[df.columns[-1]].values)
print(df[df.columns[-1]].values)
s = model.score(df[columns[:-1]].values,df[df.columns[-1]].values)
print (s)


Age = 40
Sex = 1
ChestPainType = 3
RestingBP = 140
Cholesterol = 289
FastingBS = 0
RestingECG = 3
MaxHR = 172
ExerciseAngina = 1
Oldpeak = 0.0
ST_Slope = 3

df1 = pd.DataFrame().from_dict([[Age],[Sex],[ChestPainType],[RestingBP],[Cholesterol],[RestingECG],[MaxHR],[ExerciseAngina],[Oldpeak],[ST_Slope]])
        
        
print (df1)
        
##output = model.predict([[Age,Sex,ChestPainType,RestingBP,Cholesterol,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])
output = model.predict([[40,1,3,140,289,0,3,172,1,0.0,3]])

output = model.predict(df1)
##output = model.predict(df1)
print ("-------------------",output[0])
##
##rdict = {columns[-1]: str(output[0])}
##print (rdict)
