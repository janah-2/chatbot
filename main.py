import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tflearn
import random
import json
import tensorflow
import numpy
import pickle

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?!.,"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(value) for value in doc]

    for word in words:
        if word in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output),f) 

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)


model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for word in s_words:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1
    
    return numpy.array(bag)

def compare_answers(input_string, n):
    results = model.predict([bag_of_words(input_string, words)])[0]
    results_index = numpy.argmax(results)
    result_tag = labels[results_index]

    if results[results_index] > 0.7 and "practice_question_" in result_tag:
        for tag in data["intents"]:
            if tag["tag"] == result_tag:
                responses = tag["responses"]
                print("\nGood job, another example: " + random.choice(responses) + "\n")
    else:
        if n == 1:
            for tag in data["intents"]:
                if tag["tag"] == "practice_question_1":
                    response = tag["responses"]
            print("\nI didn't get that, try saying: " + random.choice(response) + "\n")
        elif n == 2:
            for tag in data["intents"]:
                if tag["tag"] == "practice_question_2":
                    response = tag["responses"]
            print("\nI didn't get that, try saying: " + random.choice(response) + "\n")
        elif n == 3:
            for tag in data["intents"]:
                if tag["tag"] == "practice_question_3":
                    response = tag["responses"]
            print("\nI didn't get that, try saying: " + random.choice(response) + "\n")
        elif n == 4:
            for tag in data["intents"]:
                if tag["tag"] == "practice_question_4":
                    response = tag["responses"]
            print("\nI didn't get that, try saying: " + random.choice(response) + "\n")
        elif n == 5:
            for tag in data["intents"]:
                if tag["tag"] == "practice_question_5":
                    response = tag["responses"]
            print("\nI didn't get that, try saying: " + random.choice(response) + "\n")
        elif n == 6:
            for tag in data["intents"]:
                if tag["tag"] == "practice_question_6":
                    response = tag["responses"]
            print("\nI didn't get that, try saying: " + random.choice(response) + "\n")
        elif n == 7:
            for tag in data["intents"]:
                if tag["tag"] == "practice_question_7":
                    response = tag["responses"]
            print("\nI didn't get that, try saying: " + random.choice(response) + "\n")


def chat():
    print("This bot will help you to learn how to answer the most popular job interview questions, with some sample responses.\n")
    name = input("What's your name? ")
    print("Hello " + name + "!")
    print("\nStart talking with the bot! (Type 'quit' to stop and 'practice round' to try a practice interview.)")
    print("You can start asking the bot questions that you may be asked in an interview for some answers. If you don't know what questions you may be asked, ask the bot!\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "practice round":
            print("\nThis is the practice round! Pretend I am an interviewer and you are the interviewee. I will start asking the questions now.\n")
            practice_input_1 = input("1. What are you passionate about? Please start your response with: 'I am passionate about'... ")
            compare_answers(practice_input_1, 1)
            practice_input_2 = input("2. Why do you want to work here? Please start your response with: 'I want to work here because'... ")
            compare_answers(practice_input_2, 2)
            practice_input_3 = input("3. What motivates you? Please start your response with: 'What motivates me is'... ")
            compare_answers(practice_input_3, 3)
            practice_input_4 = input("4. What are your strengths? Please start your response with: 'One of my strengths is'... ")
            compare_answers(practice_input_4, 4)
            practice_input_5 = input("5. What are your weaknesses? Please start your response with: 'One of my weaknesses is'... ")
            compare_answers(practice_input_5, 5)
            practice_input_6 = input("6. What are your goals for the future? Please start your response with: 'My goal for the future is'... ")
            compare_answers(practice_input_6, 6)
            practice_input_7 = input("7. Where do you see yourself in five years? Please start your response with: 'In five years, I see myself'... ")
            compare_answers(practice_input_7, 7)
            print("\nThis is the end of the practice round! Good Job! Type 'practice round' to go again, 'quit' to stop speaking with the bot, or ask the bot questions you can be asked during an interview for sample responses.\n")


        else:
            results = model.predict([bag_of_words(user_input, words)])[0]
            results_index = numpy.argmax(results)
            result_tag = labels[results_index]
            
            if results[results_index] > 0.7:
                for tag in data["intents"]:
                    if tag["tag"] == result_tag:
                        responses = tag["responses"]

                print("\n" + random.choice(responses) + "\n")

            else:
                print("\nI didn't get that, can you please say that again?\n")
chat()