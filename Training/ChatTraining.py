#ToDo: please find this block of code and change api bot to your own use
# ------------------------------------------------------------------------------------------------
# Token: final = "Your Bot Token"
# Bot_Username: final = "@your bot username"
# ------------------------------------------------------------------------------------------------

import json
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from khmernltk import word_tokenize
from typing import final

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

# Load and preprocess data from intents.json
with open('../Data/data_intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
ignore_words = ['?', '.', '!']

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

all_words = [w for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

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

# Create a dataset
class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Hyperparameters
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000
batch_size = 8

# Create DataLoader
dataset = ChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save trained model
torch.save(model.state_dict(), '../data.pth')
print('Training complete. Model saved.')

# Evaluation
model.eval()
total = 0
correct = 0
with torch.no_grad():
    for inputs, labels in DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False):
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Validation Accuracy: {accuracy * 100:.2f}%')

# Save the trained model and parameters
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "./data.pth"
torch.save(data, FILE)

# Load data and initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Telegram Bot
Token: final = "Your Bot Token"
Bot_Username: final = "@your bot username"

# Function to handle the /start command
def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    update.message.reply_text("Thanks for chatting with me! I am a banana!")

# Function to handle the /custom command
def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    update.message.reply_text("This is a custom command!")

# Function to handle regular messages
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message_type = update.message.chat.type
    text = update.message.text

    print(f'User ({update.message.chat.id}) in {message_type}:"{text}"')

    if message_type == 'group':
        if Bot_Username in text:
            new_text = text.replace(Bot_Username, '').strip()
            response = handle_response(new_text)
        else:
            return
    else:
        response = handle_response(text)
    print('Bot', response)
    await update.message.reply_text(response)

# Function to handle response
def handle_response(text: str) -> str:
    processed = text.lower()
    user_input = tokenize(processed)
    X = bag_of_words(user_input, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob.item())
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
                response = response.replace('\n', ' ')
                print(response)
                return response
    else:
        return "សុំទោសផង ប្អូនអាចឆ្លើយបានតែ Contact ណាដែលទាក់ទងជាមួយ កាហ្វេ តែប៉ុណ្ណោះ ..."

# Function to handle errors
async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f'Update {update} caused error {context.error}')

if __name__ == "__main__":
    print("Starting message")
    app = Application.builder().token(Token).build()

    app.add_handler(CommandHandler('start', start_command))
    app.add_handler(CommandHandler('custom', custom_command))

    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_error_handler(error)

    print("polling")
    app.run_polling(poll_interval=3)
