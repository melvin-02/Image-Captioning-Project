from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np



def encode_image(image_path, model):
    image = load_img(image_path, target_size=(299, 299))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image)
    return feature


def generate_caption(model, photo_feature, tokenizer):
    max_seqlen = 34
    text = "startseq"

    for i in range(max_seqlen):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=max_seqlen)
        prediction = model.predict([photo_feature, sequence])
        prediction = np.argmax(prediction)
        word = tokenizer.index_word[prediction]
        text += ' ' + word
        if word == "endseq":
            break

    final = text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final.capitalize()
