from flask import Flask, request, jsonify
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from flask import Flask, request, render_template

import os
import pickle
import numpy as np
#from tqdm.notebook import tqdm
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
# load features from pickle
app = Flask(__name__)
Work_dir = "C:/Users/Brunda/Desktop/ImageCaptioning/ImageCaptioning"
DIR = "C:/Users/Brunda/Desktop/ImageCaptioning/ImageCaptioning"
with open(os.path.join(Work_dir, 'Image_features_Vgg16.pkl'), 'rb') as f:
    features = pickle.load(f)
    
    
with open(os.path.join(DIR, 'captions.txt'), 'r') as f:
    next(f)
    captions_doc = f.read()
    

#  Since it is having different captions for single Image ,create mapping of image to captions
mapping = {}
# process lines
for line in captions_doc.split('\n'):
    # split the line by comma(,) bcoz contains comma to seperete image name and captions
    tokens = line.split(',')
    if len(line) < 2:
        continue
    image_id, caption = tokens[0], tokens[1:]
    # remove extension ().jpg)from image ID 
    image_id = image_id.split('.')[0]
    # convert caption list to string
    caption = " ".join(caption)
    # create list if needed
    if image_id not in mapping:
        mapping[image_id] = []
    # store the caption
    mapping[image_id].append(caption)
    
##defining the function for Preprocessing of text data
def clean(mapping):
    for key, captions in mapping.items():
        for i in range(len(captions)):
            # take one caption at a time
            caption = captions[i]
            # preprocessing steps
            # convert to lowercase
            caption = caption.lower()
            # delete digits, special chars, etc., 
            caption = caption.replace('[^A-Za-z]', '')
            # delete additional spaces
            caption = caption.replace('\s+', ' ')
            # add start and end tags to the caption
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

                
#taking all captions in a list
all_captions = []
for key in mapping:
    for caption in mapping[key]:
        all_captions.append(caption)

# tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = 37

# Load the saved model
model = tf.keras.models.load_model('C:/Users/Brunda/Desktop/ImageCaptioning/ImageCaptioning/best_model_new.h5')



@app.route('/')
def index():
    return render_template('index.html')
@app.route('/generate-caption', methods=['POST'])
def generate_caption():
    # Get the image file from the request
    file = request.files['image']
    image = Image.open(file)

    # Load the image
    DIR = "C:/Users/Brunda/Desktop/ImageCaptioning/ImageCaptioning"
    image_id = file.filename.split('.')[0]
    img_path = os.path.join(DIR, "Images", file.filename)

    # Assuming you have defined the necessary variables such as 'model', 'features', 'tokenizer', 'max_length', and 'mapping'.

    captions = mapping[image_id]
    actual_captions = [caption for caption in captions]

    # Predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)

    # Return the captions as a JSON response
    response = {
        #'actual_captions': actual_captions,
        'predicted_caption': y_pred
    }
    return jsonify(response)

def predict_caption(model, image, tokenizer, max_length):
    # Add start tag for the generation process
    in_text = 'startseq'

    # Iterate over the max length of the sequence
    for _ in range(max_length):
        # Encode the input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence
        sequence = pad_sequences([sequence], maxlen=max_length)

        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)
        # Get the index with high probability
        yhat = np.argmax(yhat)
        # Convert the index to a word
        word = idx_to_word(yhat, tokenizer)

        # Stop if the word is not found
        if word is None:
            break

        # Append the word as input for generating the next word
        in_text += " " + word

        # Stop if we reach the end tag
        if word == 'endseq':
            break

    return in_text

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

#generate_caption()
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
