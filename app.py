from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from PIL import Image, ImageOps
import io
import os
import sqlite3
import json

app = Flask(__name__)
CORS(app)

DB_PATH = 'digits.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS digits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label INTEGER NOT NULL,
        image TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.commit()
    conn.close()

init_db()

print("Loading classification model...")
model_path = 'model.keras'
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("Classification model loaded successfully.")
else:
    print(f"Error: Could not find {model_path}")
    model = None

print("Loading generator model...")
gen_model_path = 'generator.keras'
if os.path.exists(gen_model_path):
    try:
        generator = tf.keras.models.load_model(gen_model_path)
        print("Generator loaded successfully.")
    except Exception as e:
        print(f"Error loading generator: {e}")
        generator = None
else:
    print(f"Error: Could not find {gen_model_path}")
    generator = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'alive',
        'classifier_loaded': model is not None,
        'generator_loaded': generator is not None
    })

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('L')
    img_array = np.array(image)
    corner_mean = np.mean([img_array[0, 0], img_array[0, -1], img_array[-1, 0], img_array[-1, -1]])
    if corner_mean > 127:
        image = ImageOps.invert(image)
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)
        padding = max(image.size) // 4
        image = ImageOps.expand(image, border=padding, fill=0)
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Classification model is not loaded.'}), 500
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided.'}), 400
        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        processed_image = preprocess_image(image_bytes)
        predictions = model.predict(processed_image)
        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        return jsonify({'prediction': predicted_digit, 'confidence': confidence})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['GET'])
def generate():
    if generator is None:
        return jsonify({'error': 'Generator model is not loaded.'}), 500
    try:
        label = request.args.get('label')
        if label is not None and label != "":
            label = int(label)
        else:
            label = np.random.randint(0, 10)
        latent_dim = 100
        noise = np.random.normal(0, 1, (1, latent_dim))
        label_input = np.array([[label]])
        gen_img = generator.predict([noise, label_input])
        gen_img = 0.5 * gen_img + 0.5
        gen_img = gen_img * 255.0
        img_array = gen_img[0, :, :, 0].astype(np.uint8)
        img = Image.fromarray(img_array, mode='L')
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return jsonify({
            'image': f'data:image/png;base64,{img_base64}',
            'label': label
        })
    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save', methods=['POST'])
def save_digit():
    try:
        data = request.json
        if not data or 'image' not in data or 'label' not in data:
            return jsonify({'error': 'Missing image or label'}), 400
        label = int(data['label'])
        image = data['image']
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT INTO digits (label, image) VALUES (?, ?)', (label, image))
        conn.commit()
        new_id = c.lastrowid
        conn.close()
        return jsonify({'success': True, 'id': new_id})
    except Exception as e:
        print(f"Error saving digit: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/gallery', methods=['GET'])
def get_gallery():
    try:
        label = request.args.get('label')
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        if label is not None and label != "":
            c.execute('SELECT id, label, image, created_at FROM digits WHERE label=? ORDER BY created_at DESC LIMIT 20', (int(label),))
        else:
            c.execute('SELECT id, label, image, created_at FROM digits ORDER BY created_at DESC LIMIT 50')
        rows = c.fetchall()
        conn.close()
        results = [{'id': r[0], 'label': r[1], 'image': r[2], 'created_at': r[3]} for r in rows]
        return jsonify(results)
    except Exception as e:
        print(f"Error fetching gallery: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/gallery/counts', methods=['GET'])
def get_counts():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT label, COUNT(*) as cnt FROM digits GROUP BY label')
        rows = c.fetchall()
        conn.close()
        counts = {str(r[0]): r[1] for r in rows}
        return jsonify(counts)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gallery/delete/<int:item_id>', methods=['DELETE'])
def delete_digit(item_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM digits WHERE id=?', (item_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
