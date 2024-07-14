from flask import Flask, render_template,request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import numpy as np
app = Flask(__name__)
model = load_model('Garbage1.h5')
def model_predict(path,model):
    img = image.load_img(path,target_size = (224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis = 0)
    preds = model.predict(x)
    return preds
@app.route('/',methods = ['GET'])
def home():
    return render_template('index.html')
@app.route('/predict',methods = ['POST','GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        a = np.argmax(model_predict(file_path,model))
        categories = ['cardboard','glass','metal','paper','plastic','trash']
        return render_template('index.html',prediction = str(categories[a]),show_predict = "true")
    
if __name__ == '__main__':
    app.run(debug=True)