import nltk

nltk.download('popular')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model

model = load_model('/Users/harshkesarwani/Desktop/Test/proh/model.h5')
import json
import random

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector

def get_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")

Language.factory("language_detector", func=get_lang_detector)

nlp.add_pipe('language_detector', last=True)

intents = json.loads(open('/Users/harshkesarwani/Desktop/Test/dataset/intents.json').read())
words = pickle.load(open('/Users/harshkesarwani/Desktop/Test/proh/texts.pkl', 'rb'))
classes = pickle.load(open('/Users/harshkesarwani/Desktop/Test/proh/labels.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result
    else:
        return "Sorry, I didn't understand that."


def chatbot_response(msg):
    res = getResponse(predict_class(msg, model), intents)
    return res


from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print("get_bot_response: " + userText)

    chatbot_response_text = chatbot_response(userText)
    print("chatbot_response: ", chatbot_response_text)

    return chatbot_response_text


if __name__ == "__main__":
    app.run()

#pip install spacy-langdetect==0.1.2
#python -m spacy download en_core_web_sm
