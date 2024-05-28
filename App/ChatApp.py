#ToDo: please find this block of code and change api bot to your own use
# ------------------------------------------------------------------------------------------------
# Token: final = "Your Bot Token"
# Bot_Username: final = "@your bot username"
# ------------------------------------------------------------------------------------------------

from typing import final
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes


import json
import numpy as np
from khmernltk import word_tokenize

# Tokenization function for Khmer text
def tokenize(sentence):
    return word_tokenize(sentence)

# Bag of words creation function
def bag_of_words(tokenized_sentence, words):
    sentence_words = [word for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

# Loading and preprocessing data from intents.json
with open('../Data/data_intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)


# Data preparation
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [w for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
print(all_words)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)       

X_train = np.array(X_train)
y_train = np.array(y_train)
print(X_train)
print(y_train)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create a simple dataset
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

# Loading data and initializing model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('../Data/data_intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)


FILE = "./data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = SimpleNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


import torch
import random
import json

Token: final = "Your Bot Token"
Bot_Username: final = "@your bot username"

# Function to handle the /start command
def start_command(update, context: ContextTypes.DEFAULT_TYPE):
    update.message.reply_text("Thanks for chatting with me! I am a banana!")

# Function to handle regular messages
def handle_message(update, context: ContextTypes.DEFAULT_TYPE):
    update.message.reply_text("I am a banana! Please type something so I can respond!")

# Function to handle a custom command, e.g., /custom_command
def custom_command(update, context: ContextTypes.DEFAULT_TYPE):
    update.message.reply_text("This is a custom command!")


def handle_response(text: str) -> str:
    processed = text.lower()
    # print(type(processed))
    user_input = processed
    user_input = tokenize(user_input)
    X = bag_of_words(user_input, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob.item() )
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                response = response.replace('\n', ' ')  # Replace line breaks with spaces
                print(response)
                return response
    else:
        return "I don't understand ..."

async def handle_message(update : Update, context: ContextTypes.DEFAULT_TYPE):
    message_type:str = update.message.chat.type
    text: str = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}:"{text}"')

    if message_type == 'group':
        if Bot_Username in text:
            new_text : str = text.replace(Bot_Username, '').strip()
            response: str = handle_response(new_text)
        else:
            return
    else:
        response: str = handle_response(text)
    print('Bot', response)
    await update.message.reply_text(response)

async def error(update : Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == "__main__":
    print("Starting messaage")
    app = Application.builder().token(Token).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('help', help))
    app.add_handler(CommandHandler('custom', custom_command))

    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_error_handler(error)

    print("polling")
    app.run_polling(poll_interval=3)