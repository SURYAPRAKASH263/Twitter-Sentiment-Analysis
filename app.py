import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy
import numpy as np

app=Flask(__name__)

## Load the model

sentiment_model=pickle.load(open('sentiment_model.pkl','rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])

def predict_api():

    data =[float(x) for x in request.form.values()]
    final_input=vectorizer.tarnsform(np.array(data).reshape(1,-1))
    print(final_input)
    output=vectorizer.predict(final_input)[0]
    return render_template("home.html",prediction_text="The Sentiment of tweet is {}".format(output))

    
    
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=vectorizer.transform(np.array(list(data.values())).reshape(1,-1))
    output=sentiment_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

if __name__=="__main__":
    app.run(debug=True)