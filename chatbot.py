import json
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import pickle
import os
from flask import Flask, render_template, request, jsonify
import re
from datetime import datetime
import threading
import time

# --- NLTK Data Download Check and Installation ---
try:
    nltk.data.find('corpora/wordnet.zip')
    nltk.data.find('tokenizers/punkt.zip')
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

class AdvancedChatbot:
    def __init__(self):
        self.user_context = {}
        self.conversation_history = []
        self.user_emotions = []
        self.mood_patterns = {
            'positive': ['happy', 'good', 'great', 'awesome', 'excited', 'khush', 'accha'],
            'negative': ['sad', 'angry', 'frustrated', 'upset', 'dukhi', 'pareshan', 'gussa'],
            'anxious': ['worried', 'nervous', 'anxious', 'stressed', 'tension', 'chinta']
        }
        self.load_or_train_model()
        
    def load_or_train_model(self):
        if not os.path.exists('advanced_chatbot_model.h5') or not os.path.exists('words.pkl') or not os.path.exists('classes.pkl'):
            print("Training advanced model...")
            self.train_model()
        else:
            print("Loading existing model...")
            self.load_model()
    
    def train_model(self):
        words = []
        classes = []
        documents = []
        ignore_words = ['?', '!', '.', ',', "'s", "'re", "'m", "'ll", "'ve"]
        
        with open('advanced_intents.json', 'r', encoding='utf-8') as file:
            intents = json.load(file)
        
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                word_list = nltk.word_tokenize(pattern.lower())
                words.extend(word_list)
                documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
        
        words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_words]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        
        training = []
        output_empty = [0] * len(classes)
        
        for doc in documents:
            bag = []
            pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0] if word not in ignore_words]
            
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1
            training.append([bag, output_row])
        
        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])
        
        model = Sequential()
        model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(train_y[0]), activation='softmax'))
        
        adam = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        
        model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=8, verbose=1)
        
        model.save('advanced_chatbot_model.h5')
        with open('words.pkl', 'wb') as file:
            pickle.dump(words, file)
        with open('classes.pkl', 'wb') as file:
            pickle.dump(classes, file)
        
        self.model = model
        self.words = words
        self.classes = classes
    
    def load_model(self):
        self.model = load_model('advanced_chatbot_model.h5')
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
    
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words
    
    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)
    
    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.20
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list
    
    def analyze_emotion(self, text):
        text_lower = text.lower()
        detected_emotion = 'neutral'
        confidence = 0.0
        
        for emotion, keywords in self.mood_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            current_confidence = matches / len(keywords)
            if current_confidence > confidence:
                confidence = current_confidence
                detected_emotion = emotion
        
        return detected_emotion, confidence
    
    def generate_contextual_response(self, intents_list, user_message, user_id="default"):
        if not intents_list:
            return self.handle_unknown_input(user_message)
        
        tag = intents_list[0]['intent']
        confidence = float(intents_list[0]['probability'])
        
        emotion, emotion_confidence = self.analyze_emotion(user_message)
        
        if user_id not in self.user_context:
            self.user_context[user_id] = {
                'name': None,
                'last_emotion': emotion,
                'conversation_count': 0,
                'topics_discussed': []
            }
        
        self.user_context[user_id]['conversation_count'] += 1
        self.user_context[user_id]['last_emotion'] = emotion
        
        if tag not in self.user_context[user_id]['topics_discussed']:
            self.user_context[user_id]['topics_discussed'].append(tag)
        
        with open('advanced_intents.json', 'r', encoding='utf-8') as file:
            intents = json.load(file)
        
        for intent in intents['intents']:
            if intent['tag'] == tag:
                responses = intent['responses']
                
                if emotion == 'negative' and 'supportive_responses' in intent:
                    responses.extend(intent['supportive_responses'])
                elif emotion == 'positive' and 'celebratory_responses' in intent:
                    responses.extend(intent['celebratory_responses'])
                
                response = random.choice(responses)
                
                if self.user_context[user_id]['name']:
                    response = response.replace('[NAME]', self.user_context[user_id]['name'])
                else:
                    response = response.replace('[NAME]', 'friend')
                
                if confidence > 0.8 and random.random() > 0.6:
                    followups = self.get_followup_questions(tag, emotion)
                    if followups:
                        response += " " + random.choice(followups)
                
                return response
        
        return self.handle_unknown_input(user_message)
    
    def get_followup_questions(self, tag, emotion):
        followups = {
            'feeling_sad': [
                "Would you like to talk about what's bothering you?",
                "Sometimes sharing helps. What's on your mind?",
                "I'm here to listen. Want to tell me more?"
            ],
            'greeting': [
                "What brings you here today?",
                "How has your day been so far?",
                "What's on your mind today?"
            ],
            'ask_for_coping_strategy': [
                "Have you tried any relaxation techniques before?",
                "What usually helps you feel better?",
                "Would you like me to guide you through a quick exercise?"
            ]
        }
        return followups.get(tag, [])
    
    def handle_unknown_input(self, user_message):
        if '?' in user_message:
            return "That's an interesting question. While I'm still learning, I'd love to help you explore that topic. Can you tell me more about what you're looking for?"
        elif any(word in user_message.lower() for word in ['help', 'madad', 'support']):
            return "I'm here to support you. Can you help me understand what kind of help you're looking for? Whether it's emotional support, coping strategies, or just someone to listen."
        else:
            responses = [
                "I find that fascinating. Can you tell me more about that?",
                "That's interesting. I'm still learning about such topics. What would you like to explore about this?",
                "I hear you. Sometimes the most important conversations start with topics I'm still learning about. What matters most to you about this?",
                "Hmm, that's something new for me. I'd love to understand better - can you share more of your thoughts on this?"
            ]
            return random.choice(responses)
    
    def extract_name(self, message):
        patterns = [
            r"my name is (\w+)",
            r"i am (\w+)",
            r"call me (\w+)",
            r"mera naam (\w+) hai"
        ]
        for pattern in patterns:
            match = re.search(pattern, message.lower())
            if match:
                return match.group(1).capitalize()
        return None

chatbot = AdvancedChatbot()

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("advanced_index.html")

@app.route("/get")
def get_bot_response():
    user_message = request.args.get('msg')
    user_id = request.args.get('user_id', 'default')
    
    if not user_message:
        return "Please say something!"
    
    crisis_keywords = [
        "end my life", "harm myself", "suicidal", "kill myself", "want to die",
        "jaan deni", "nuksaan pahunchana", "marne ka man hai", "jina nahi chahta",
        "suicide", "ending it all", "can't go on", "no point living"
    ]
    
    if any(keyword in user_message.lower() for keyword in crisis_keywords):
        with open('advanced_intents.json', 'r', encoding='utf-8') as file:
            intents = json.load(file)
        for intent in intents['intents']:
            if intent['tag'] == 'crisis_support':
                return random.choice(intent['responses'])
    
    name = chatbot.extract_name(user_message)
    if name and user_id not in chatbot.user_context:
        chatbot.user_context[user_id] = {'name': name, 'last_emotion': 'neutral', 'conversation_count': 0, 'topics_discussed': []}
        return f"Nice to meet you, {name}! I'm onepersonai, and I'm here to support you. How are you feeling today?"
    elif name and user_id in chatbot.user_context:
        chatbot.user_context[user_id]['name'] = name
        return f"Got it, I'll call you {name}! How can I help you today?"
    
    predicted_intents = chatbot.predict_class(user_message)
    response = chatbot.generate_contextual_response(predicted_intents, user_message, user_id)
    
    return response

@app.route("/mood_analysis")
def analyze_user_mood():
    user_id = request.args.get('user_id', 'default')
    if user_id in chatbot.user_context:
        context = chatbot.user_context[user_id]
        return jsonify({
            'current_emotion': context.get('last_emotion', 'neutral'),
            'conversation_count': context.get('conversation_count', 0),
            'topics_discussed': context.get('topics_discussed', [])
        })
    return jsonify({'current_emotion': 'neutral', 'conversation_count': 0, 'topics_discussed': []})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)