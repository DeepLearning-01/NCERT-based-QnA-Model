import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('physics_tb.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details = False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key = lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, file_json):
    tag = ints[0]['intent']
    lists = file_json['physics_tb']
    for i in lists:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


#Creating GUI with tkinter
import tkinter
from tkinter import *
from time import sleep

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state = NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="red", font = ("Arial", 14, 'bold' ))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Tanya: " + res + '\n\n')
            
        ChatLog.config(state = DISABLED)
        ChatLog.yview(END)
        if res in ["See you!", "Have a nice day", "Bye! Come back again soon."]:
            sleep(2)
            window.destroy()

window = Tk()
window.title("Physics is exotic")

window.geometry("1000x1000")
window.resizable(width = False, height = False)

#Create Chat window
ChatLog = Text(window, bd = 2, bg = "white", height = "12", width = "500", font = ("Arial", 14, 'bold'), fg = "red")

ChatLog.config(state = DISABLED)

#Bind scrollbar to Chat window
scrollbar = Scrollbar(window, command=ChatLog.yview, cursor="arrow")
ChatLog['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(window, font = ("Verdana",14,'bold'), text = "Ask", width = "10", height = 5,
                    bd = 0, bg = "#ffff00", activebackground = "#3c9d9b",fg = '#ff0000',
                    command = send )	

#Create the box to enter message
EntryBox = Text(window, bd = 0, bg = "white",width = "950", height = "150", font = ("Arial", 14, "bold"))
EntryBox.config(state = NORMAL)
#EntryBox.bind("<Return>", send)


#Place all components on the screen
scrollbar.place(x = 985,y = 6, height = 886)
ChatLog.place(x = 6,y = 6, height = 886, width = 980)
EntryBox.place(x = 10, y = 905, height = 80, width = 840)
SendButton.place(x = 850, y = 905, height = 80)

window.mainloop()
