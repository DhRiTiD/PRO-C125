from classifier import get_prediction
from flask import Flask, jsonify, request

app = Flask(__name__) #invoking constructor

@app.route('/predict-alpha', methods = ['POST'])

def predict_alpha():

    img = request.files.get('alpha')
    prediction = get_prediction(img)

    return(jsonify({
        'prediction' : prediction,
       }), 200)

if __name__ == '__main__':
    app.run(debug=True, port=9090)
    