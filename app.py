import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
import os
import pickle
from flask import Flask, redirect, request, url_for, flash, render_template, send_from_directory
from werkzeug.utils import secure_filename
from caption import encode_image, generate_caption

inception_model = InceptionV3()
inception_model = Model(inception_model.inputs,
                        inception_model.layers[-2].output)

UPLOAD_FOLDER = r"E:\pythonNotebooks\Image captioning\upload_folder"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

caption_model = tf.keras.models.load_model('caption_model_final.h5')
with open('tokenizer_2000.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/',  methods=['GET', 'POST'])
def upload_image():
    if request.method == "POST":

        if 'file' not in request.files:
            flash('No file found')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            image_feature = encode_image(os.path.join(
                app.config['UPLOAD_FOLDER'], filename), inception_model)
            caption = generate_caption(caption_model, image_feature, tokenizer)
            return render_template('result.html', filename=filename, caption=caption)

    return render_template('index.html')


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
    app.run(debug=True)
