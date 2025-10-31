import joblib
import pandas as pd
import numpy as np
import random
from flask import Flask, request, jsonify, render_template
import tensorflow as tf


catenc = joblib.load("catenc.pkl")
cityenc = joblib.load("cityenc.pkl")
monthenc = joblib.load("monthenc.pkl")
regenc = joblib.load("regenc.pkl")
subenc = joblib.load("subenc.pkl")
yrenc = joblib.load("yrenc.pkl")
rfmodel=joblib.load("Random_Forest_Regression.pkl")
ann = tf.keras.models.load_model("ann_model.keras")

app = Flask(__name__)

categories=['Snacks', 'Eggs, Meat & Fish', 'Fruits & Veggies', 'Bakery',
       'Beverages', 'Food Grains', 'Oil & Masala']
subcat=['Health Drinks', 'Soft Drinks', 'Cookies', 'Breads & Buns',
       'Chocolates', 'Noodles', 'Masalas', 'Biscuits', 'Cakes',
       'Edible Oil & Ghee', 'Spices', 'Mutton', 'Eggs', 'Organic Staples',
       'Fresh Fruits', 'Fish', 'Fresh Vegetables', 'Atta & Flour',
       'Organic Fruits', 'Chicken', 'Organic Vegetables', 'Dals & Pulses',
       'Rice']
cities=['Kanyakumari', 'Tirunelveli', 'Bodi', 'Krishnagiri', 'Vellore',
       'Perambalur', 'Tenkasi', 'Chennai', 'Salem', 'Karur', 'Pudukottai',
       'Coimbatore', 'Ramanadhapuram', 'Cumbum', 'Virudhunagar', 'Madurai',
       'Ooty', 'Namakkal', 'Viluppuram', 'Dindigul', 'Theni', 'Dharmapuri',
       'Nagercoil', 'Trichy']
months=[11, 12, 9, 10, 5, 6, 7, 8, 3, 4, 1, 2]
years=[2018, 2017, 2016, 2015]
regions=['West', 'East', 'Central', 'South', 'North']
def predict(cat1,sub1,city1,yr1,month1,reg1,disc1,profit1):
    cat=cat1
    sub=sub1
    city=city1
    yr=yr1
    month=month1
    reg=reg1
    disc=disc1
    profit=profit1
    print(cat,sub,city,yr,month,reg,disc,profit)
    data = []
    data.append(disc);
    data.append(profit);
    c=catenc.transform(pd.DataFrame({'Category':[cat]}))
    sc=subenc.transform(pd.DataFrame({'Sub Category':[sub]}))
    mnth=monthenc.transform(pd.DataFrame({'month':[month]}))
    regd=regenc.transform(pd.DataFrame({'Region':[reg]}))
    cityd=cityenc.transform(pd.DataFrame({'City':[city]}))
    yrd=yrenc.transform(pd.DataFrame({'year':[yr]}))
    data=data+list(c[0])
    data=data+list(cityd[0])
    data=data+list(sc[0])
    data=data+list(regd[0])
    data=data+list(yrd[0])
    data=data+list(mnth[0])
    newData=[data]
    data=np.array(newData)
    rfpred = rfmodel.predict(data)
    annpred = ann.predict(data)
    predicted_value=(rfpred[0]+annpred[0])/2
    return predicted_value

@app.route('/')
def home():
    return render_template('index.html',status=True)

@app.route('/predict', methods=['POST'])
def prediction():
    features=[x for x in request.form.values()]
    answer = predict(features[5],features[6],features[7],int(features[2]),int(features[3]),features[4],float(int(features[0])/100),float(features[1]))
    up=answer[0]-387.9304
    down=answer[0]+387.9304
    if up<0:
        up=0
    return render_template('index.html',status=False,up=round(up,3),down=round(down,3),ans=round(answer[0],3))

if __name__ == '__main__':
    app.run(debug=True)


# 