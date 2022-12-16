from typing import Union

from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()




@app.get("/predict")
def read_item(age:float,bmi:float,children:float,sex_embedding_0:float,smoker_embedding_0:float,region_embedding_0:float,region_embedding_1:float):
    model = tf.keras.models.load_model('modelo_regresion')
    model.summary()
    predict = model.predict([age,bmi,children,sex_embedding_0,smoker_embedding_0,region_embedding_0,region_embedding_1]).flatten()
    print(predict)
    return {"predict": float(predict)}

