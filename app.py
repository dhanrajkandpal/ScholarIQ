from flask import flask , request , render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
application = flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        return render_template('home.html')
    else:
        pass