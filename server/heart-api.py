import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LinearRegression, LogisticRegression
from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import json

##
## training models
##

#read and train the data set
df = pd.read_csv("heart.csv")

#get non numeric values columns
dataFrameColums = df.columns
numericCols = df.select_dtypes([np.number]).columns

#create json object to build a form
data = {}
dataSet = df
dataSet.drop(labels= dataFrameColums[-1], axis=1)
for c in dataFrameColums:
        if c not in numericCols:
            arr = dataSet[c].to_numpy()
            arr  = list(set(arr))   
            if (len(arr) == 2):
                data[c] ={"name":c,"data" : arr,"type" : "radio"}
                data[c]
            elif (len(arr) < 10):
                data[c] ={"name":c,"data" : arr,"type" : "select"}
                data[c] 
            else :
                data[c] ={"name":c,"data" : arr,"type" : "text"}
                data[c] 
        else :
            data[c] ={"name":c,"type" : "number"}
            data[c] 

json_data = json.dumps(data)        
json_data = json.loads(json_data)  


#get non numeric colums
nonNumericCols = list((Counter(dataFrameColums) - Counter(numericCols)).elements())

#changing non numeric values colums in to numaric columns
le = LabelEncoder()
for c in nonNumericCols:
    df[c] = le.fit_transform(df[c])

columns = list(df.columns)

#logisticcRegrassion model training
logisticRegressionModel = LogisticRegression()
logisticRegressionModel = LogisticRegression(solver='lbfgs', max_iter=len(df))
logisticRegressionModel.fit(df[columns[:-1]],df[df.columns[-1]])

#LinearRegression model training
linearRegressionModel = LinearRegression()
linearRegressionModel.fit(df[columns[:-1]],df[df.columns[-1]])

#scores of the models
linearScore = linearRegressionModel.score(df[columns[:-1]],df[df.columns[-1]])
logisticScore = logisticRegressionModel.score(df[columns[:-1]],df[df.columns[-1]])
print ("logistic regression model score : " , logisticScore)
print ("linear regression model score : " , linearScore)

##
## backend flask server
##

#creating flask app
app = Flask(__name__)
CORS(app)
api = Api(app)

#get parameter values
parser = reqparse.RequestParser()
for c in columns[:-1]:
    parser.add_argument(c,location='form')

# post result
class Heart(Resource):
    def post(self):

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

        #create data fream using parameters
        df1 = pd.DataFrame().from_dict([args])
        #predect 
        linearRegressionOutput = linearRegressionModel.predict(df1)
        logisticRegressionOutput = logisticRegressionModel.predict(df1)
        #return predetion
        if(linearScore > logisticScore):
            rdict = {columns[-1]: str(linearRegressionOutput[0]),"args":args}
        else :
            rdict = {columns[-1]: str(logisticRegressionOutput[0]),"args":args}
        
        print (rdict)
        return rdict, 200
        

# return dataset details to front end for create UI
class getHeader(Resource):
    def get(self) :

        rdict = {"data" :json_data, "columns" : dataSet.columns.values.tolist()}      

        print (rdict)
        return rdict, 200
        

##
## Api resource routing here
##
api.add_resource(Heart, '/heart')
api.add_resource(getHeader, '/getData')


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
