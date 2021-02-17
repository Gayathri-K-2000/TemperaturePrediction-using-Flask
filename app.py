from flask import Flask, jsonify,request
import tensorflow as tf
app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict(): 
    country = request.form['country']
    city = request.form['city']
    month = request.form['month']
    day = request.form['day']
    year = request.form['year']
    pre=""
    model = tf.keras.models.load_model("temperature_model.h5")

    pred = model.predict([[int(country), int(city), int(month), int(day), int(year)]])
    p=pred.tolist()
    for ele in p[0]:
        pre += str(ele)
    return jsonify(temperature= pre)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
