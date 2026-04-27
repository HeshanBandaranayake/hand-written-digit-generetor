from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

generator = tf.keras.models.load_model("generator_model.h5")

# simple in-memory database
database = {str(i): [] for i in range(10)}

@app.route('/generate', methods=['POST'])
def generate():
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)

    img = (generated_image[0] * 127.5 + 127.5).astype(int)
    img = img.reshape(28, 28)

    return jsonify(img.tolist())


@app.route('/save', methods=['POST', 'OPTIONS'])
def save():
    if request.method == 'OPTIONS':
        return '', 200

    data = request.json
    digit = str(data['digit'])
    image = data['image']

    database[digit].append(image)

    return jsonify({"message": "saved"})




@app.route('/get/<digit>', methods=['GET'])
def get_digit(digit):
    if digit not in database or len(database[digit]) == 0:
        return jsonify({"error": "no data"})

    img = database[digit][np.random.randint(0, len(database[digit]))]
    return jsonify(img)


if __name__ == '__main__':
    app.run(debug=True)