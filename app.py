# Healthcare App.py

from flask import Flask, render_template, request
import pickle
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

###############################################################################
@app.route('/')
def home():
    return render_template('home.html')
###############################################################################
    

###############################################################################
@app.route('/cancer',methods=['GET','POST'])
def cancer():
    model = pickle.load(open('cancer.pkl','rb'))
    cancer_scaler = pickle.load(open('cancer_scaler.pkl','rb'))
    try:
        if request.method == 'POST':
           radius_mean = float(request.form['radius_mean'])
           texture_mean = float(request.form['texture_mean'])
           perimeter_mean = float(request.form['perimeter_mean'])
           area_mean = float(request.form['area_mean'])
           smoothness_mean = float(request.form['smoothness_mean'])
           compactness_mean = float(request.form['compactness_mean'])
           concavity_mean = float(request.form['concavity_mean'])
           concave_points_mean = float(request.form['concave_points_mean'])
           symmetry_mean = float(request.form['symmetry_mean'])
           fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
           radius_se = float(request.form['radius_se'])
           texture_se = float(request.form['texture_se'])
           perimeter_se = float(request.form['perimeter_se'])
           area_se = float(request.form['area_se'])
           smoothness_se = float(request.form['smoothness_se'])
           compactness_se = float(request.form['compactness_se'])
           concavity_se = float(request.form['concavity_se'])
           concave_points_se = float(request.form['concave_points_se'])
           symmetry_se = float(request.form['symmetry_se'])
           fractal_dimension_se = float(request.form['fractal_dimension_se'])
           radius_worst = float(request.form['radius_worst'])
           texture_worst = float(request.form['texture_worst'])
           perimeter_worst = float(request.form['perimeter_worst'])
           area_worst = float(request.form['area_worst'])
           smoothness_worst = float(request.form['smoothness_worst'])
           compactness_worst = float(request.form['compactness_worst'])
           concavity_worst = float(request.form['concavity_worst'])
           concave_points_worst = float(request.form['concave_points_worst'])
           symmetry_worst = float(request.form['symmetry_worst'])
           fractal_dimension_worst = float(request.form['fractal_dimension_worst'])
           scaled = cancer_scaler.transform([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,
                                              concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,
                                              texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,
                                              symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,
                                              smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,
                                              fractal_dimension_worst]])
           result = model.predict(scaled)
           if result[0] == 0:
               return render_template('cancer.html',result='You have a low chance of having Cancer')
           elif result[0] == 1:
               return render_template('cancer.html',result='You have a very high chance of having Cancer')
           else:
               return render_template('home.html')
    except ValueError:
        return render_template('cancer.html',result='Invalid Input')
        
    return render_template('cancer.html')
###############################################################################
   
    
###############################################################################
@app.route('/diabetes',methods=['GET','POST'])
def diabetes():
    model = pickle.load(open('diabetes.pkl','rb'))
    try:
        if request.method == 'POST':
            preg = float(request.form['preg'])
            glucose = float(request.form['glucose'])
            bp = float(request.form['bp'])
            st = float(request.form['st'])
            ins = float(request.form['ins'])
            bmi = float(request.form['bmi'])
            dpf = float(request.form['dpf'])
            age = float(request.form['age'])
            result = model.predict([[preg,glucose,bp,st,ins,bmi,dpf,age]])
            if result[0] == 0:
               return render_template('diabetes.html',result='Congratulations! You do not have diabetes.')
            elif result[0] == 1:
               return render_template('diabetes.html',result='You have a very high chance of having Diabetes!')
            else:
               return render_template('home.html')
    except ValueError:
        return render_template('diabetes.html',result='Invalid Input')
        
    return render_template('diabetes.html')
###############################################################################
    

###############################################################################
@app.route('/heart',methods=['GET','POST'])
def heart():
    model = pickle.load(open('heart.pkl','rb'))
    try:
        if request.method == 'POST':
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            cp = float(request.form['cp'])
            trestbps = float(request.form['trestbps'])
            chol = float(request.form['chol'])
            fbs = float(request.form['fbs'])
            restecg = float(request.form['restecg'])
            thalach = float(request.form['thalach'])
            exang = float(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = float(request.form['slope'])
            ca = float(request.form['ca'])
            thal = float(request.form['thal'])
            result = model.predict([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,
                                           exang,oldpeak,slope,ca,thal]])
            if result[0] == 0:
                return render_template('heart.html',result='Congratulations! Your heart seems to be healthy!')
            elif result[0] == 1:
               return render_template('heart.html',result='Your heart is not in great shape!')
            else:
               return render_template('home.html')
    except ValueError:
        return render_template('heart.html',result='Invalid Input')
    
    return render_template('heart.html')    
###############################################################################
    

###############################################################################
@app.route('/liver',methods=['GET','POST'])
def liver():
    model = pickle.load(open('liver.pkl','rb'))
    try:
        if request.method == 'POST':
            age = float(request.form['age'])
            sex = float(request.form['sex'])
            tb = float(request.form['tb'])
            db = float(request.form['db'])
            ap = float(request.form['ap'])
            alt = float(request.form['alt'])
            ast = float(request.form['ast'])
            tp = float(request.form['tp'])
            alb = float(request.form['alb'])
            ag = float(request.form['ag'])
            result = model.predict([[age,sex,tb,db,ap,alt,ast,tp,alb,ag]])
            if result[0] == 1:
                return render_template('liver.html',result='You have a problem in your liver')
            elif result[0] == 2:
                return render_template('liver.html',result='Congratulations! Your liver is fine.')
    except ValueError:
        return render_template('liver.html',result='Invalid Input')
    
    return render_template('liver.html')
###############################################################################
    

###############################################################################
@app.route('/malaria',methods=['GET','POST'])
def malaria():
    malaria_model = tf.keras.models.load_model("malaria.h5")
    if request.method == 'POST':
        img = Image.open(request.files['image'])
        img = img.resize((64,64))
        img = np.array(img)
        img = img / 255.0
        img = img.reshape(1,64,64,3)
        result = malaria_model.predict_classes(img)[0][0]
        acc = malaria_model.predict(img)[0][0]
        
        if result == 0:
            return render_template('malaria.html',result='You have Malaria! Take care of yourself.',accuracy=f'Accuracy is {(100 - acc*100):.2f} %')
        if result == 1:
            return render_template('malaria.html',result='You are perfectly fine!',accuracy=f'Accuracy is {(acc*100):.2f} %')
        
    return render_template('malaria.html')
###############################################################################
    

###############################################################################
@app.route('/details',methods=['GET'])
def details():
    return render_template('details.html')
###############################################################################

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)