import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("heart.csv")
# df = pd.read_csv('heart.csv',dtype={'col1':str,'col2':np.int64,'col3':np.float64})

le = LabelEncoder()
for c in ['Sex','ST_Slope','ExerciseAngina', 'ChestPainType','RestingECG']:
    df[c] = le.fit_transform(df[c])



# data['ExerciseAngina_num'] = data['ExerciseAngina'].apply(ExerciseAngina_to_numeric)
# data['ChestPainType_num'] = data['ChestPainType'].apply(ChestPainType_to_numeric)
# data['gender_num'] = data['Sex'].apply(gender_to_numeric)
# data['RestingECG_num'] = data['RestingECG'].apply(RestingECG_to_numeric)
# data['ST_Slope_num'] = data['ST_Slope'].apply(ST_Slope_to_numeric)

# model = LinearRegression()
model = LogisticRegression()
# model.fit(data[['Age','gender_num','ChestPainType_num','RestingBP','Cholesterol','FastingBS','RestingECG_num','MaxHR','ExerciseAngina_num','Oldpeak','ST_Slope_num']],df.HeartDisease)
columns = list(df.columns)
model.fit(df[columns[:-1]],df[columns[-1]])

app = Flask(__name__)
api = Api(app)

s = model.score(df[columns[:-1]],df[columns[-1]])
print (s)

parser = reqparse.RequestParser()
for c in columns[:-1]:
    parser.add_argument(c)
    
# parser.add_argument('Sex')
# parser.add_argument('ChestPainType')
# parser.add_argument('RestingBP')
# parser.add_argument('Cholesterol')
# parser.add_argument('FastingBS')
# parser.add_argument('RestingECG')
# parser.add_argument('MaxHR')
# parser.add_argument('ExerciseAngina')
# parser.add_argument('Oldpeak')
# parser.add_argument('ST_Slope')

# Age = 40
# Sex = 1
# ChestPainType = 3
# RestingBP = 140
# Cholesterol = 289
# FastingBS = 0
# RestingECG = 3
# MaxHR = 172
# ExerciseAngina = 1
# Oldpeak = 0.0
# ST_Slope = 3



# Todo
# shows a single todo item and lets you delete a todo item
class Heart(Resource):
    def get(self):
        args = parser.parse_args()
        if args['Age'] is None: args['Age']=40
        if args['Sex'] is None: args['Sex']=1
        if args['ChestPainType'] is None: args['ChestPainType']=3
        if args['RestingBP'] is None: args['RestingBP']=140
        if args['Cholesterol'] is None: args['Cholesterol']=289
        if args['FastingBS'] is None: args['FastingBS']=80
        if args['RestingECG'] is None: args['RestingECG']=3
        if args['MaxHR'] is None: args['MaxHR']=170
        if args['ExerciseAngina'] is None: args['ExerciseAngina']=1
        if args['Oldpeak'] is None: args['Oldpeak']=150
        if args['ST_Slope'] is None: args['ST_Slope']=3
        
        df1 = pd.DataFrame().from_dict([args])
        
        
        print (df1)
        # Age = 40
        # Sex = 1
        # ChestPainType = 3
        # RestingBP = 140
        # Cholesterol = 289
        # FastingBS = 0
        # RestingECG = 3
        # MaxHR = 172
        # ExerciseAngina = 1
        # Oldpeak = 0.0
        # ST_Slope = 3
        output = model.predict(df1)
        # output = model.predict([[40,1,3,140,289,0,3,172,1,0.0,3]])
        print ("-------------------",output)
        # result = model.coef_[0] * Age + model.coef_[1] * Sex + model.coef_[2] * ChestPainType + model.coef_[3] * RestingBP + model.coef_[4] * Cholesterol + model.coef_[5] * FastingBS + model.coef_[6] * RestingECG + model.coef_[7] * MaxHR + model.coef_[8] * ExerciseAngina + model.coef_[9] * Oldpeak + model.coef_[10] * ST_Slope + model.coef_[5] * 0 + model.intercept_
        
        # rdict = {'HeartDisease': np.random.random()}
        rdict = {columns[-1]: str(output[0])}
        print (rdict)
        return rdict, 200
        

##
## Actually setup the Api resource routing here
##
api.add_resource(Heart, '/heart')



if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)