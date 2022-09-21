import dill
import pandas as pd
import uvicorn
from fastapi import  FastAPI
from pydantic import BaseModel
from get_columns import get_columns

import json
app = FastAPI()
with open('src/pipeline.pkl', 'rb') as file:
    model = dill.load(file)


@app.get('/status')
def status():
    return 'I m OK'

app.get('/version')
def version():
    return model['metadata']

# print(get_columns())

class Form (BaseModel):
    id: int
    url: str
    region: str
    region_url: str
    price: int
    year: float
    manufacturer: str
    model: str
    fuel: str
    odometer: float
    title_status: str
    transmission: str
    image_url: str
    description: str
    state: str
    lat: float
    long: float
    posting_date: str

class Prediction(BaseModel):
    id : int
    pred:str
    price:int

@app.post('/predict',response_model=Prediction)
def predict(form:Form):
    data = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(data)

    return {
        'id':form.id,
        'pred':y[0],
        'price':form.price
    }

if __name__ == '__main__':
    uvicorn.run(app,host='127.0.0.1',port=8000,log_level='info')