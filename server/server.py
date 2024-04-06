# server.py



# app = Flask(__name__)
# model = load_model_from_h5('..\saved_models\Emotion_Voice_Detection_Model.h5')
# model = load_model_from_h5('../saved_models/Emotion_Voice_Detection_Model.h5')




# if __name__ == '__main__':
#     app.run('4000', debug=True)
#     print("server running")


from flask import Flask, request, jsonify
from model_loader import load_model_from_h5
import tensorflow as tf
import numpy as np
import joblib
lb = joblib.load('../label_encoder.pkl')


model = tf.keras.models.load_model("../saved_models/Emotion_Voice_Detection_Model.h5")

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # if 'file' not in request.files:
    #     return jsonify({'error': 'No file part'})

    # file = request.files['file']

    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'})

    data = np.array(data)
    livepreds = model.predict(data, batch_size=32, verbose=0)
    print(livepreds)
    preds1=livepreds.argmax(axis=1)
    print(preds1)
    abc = preds1.astype(int).flatten()
    predictions = (lb.inverse_transform((abc)))
    print(predictions)
    return jsonify(livepreds.tolist())


if __name__ == '__main__':
    app.run(debug=True)