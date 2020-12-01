import pickle
import os
import pandas as pd
from fastapi import FastAPI
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#PORT = int(os.environ.get("PORT", 5000))


with open('trained_model.plk','rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
async def main():
#    userinput = {'Pregnancies':7,'Glucose':158,'BloodPressure':120,
#            'SkinThickness':25,'Insulin':200,'BMI':33.6,'Age':31}
#    data = pd.DataFrame(data=userinput)
#    prediction= model.predict(data)
    return 'Deployed'

@app.get("/predict")
async def create_item(Pregnancies:float,
                        Glucose :float,
                        BloodPressure:float,
                        SkinThickness:float,
                        Insulin:float,
                        BMI:float,
                        Age:float):

    #userinput = {'Pregnancies':int(Pregnancies),'Glucose':int(Glucose),'BloodPressure':BloodPressure,'SkinThickness':SkinThickness,'Insulin':Insulin,'BMI':BMI,'Age':Age}

    #print(userinput)

    #data = pd.DataFrame(data=userinput)

    #sc_X = StandardScaler()

    #X =  pd.DataFrame(sc_X.fit_transform(data),columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'Age'])
    data = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,Age]]

    sc_X = StandardScaler()

    X =  pd.DataFrame(sc_X.fit_transform(data),columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'Age'])

    result_outcome = model.predict(X)[0]
    result_proba = model.predict_proba(X)[0]

    #return {'OUTCOME':int(result_outcome)}
    return {"OUTCOME":int(result_outcome),"RISK_0":result_proba[0],"RISK_1":result_proba[1]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

