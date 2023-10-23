# Import necessary libraries
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeutralNet  # Import your NeutralNet model from model.py

# Load intent data from a JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Initialize empty lists to store words, tags, and (tokenized_sentence, tag) pairs
all_words = []
tags = []
xy = []

# Iterate through intents and their patterns
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)  # Tokenize the pattern
        all_words.extend(w)  # Add words to the all_words list
        xy.append((w, tag))  # Append tokenized sentence and its tag

# Define a list of words to ignore (e.g., punctuation)
ignoreWords = ['?', '!', '.', ',']

# Stem the words and remove ignored words, then sort and make them unique
all_words = [stem(w) for w in all_words if w not in ignoreWords]
all_words = sorted(set(all_words))
tags = sorted(set(tags))  # Sort and make tags unique

# Create training data
XTrain = []
yTrain = []

# Iterate through (tokenized_sentence, tag) pairs
for (pattern_sentence, tag) in xy:
    # Create a bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    XTrain.append(bag)

    # For PyTorch CrossEntropyLoss, we need class labels, not one-hot encoded labels
    label = tags.index(tag)
    yTrain.append(label)

XTrain = np.array(XTrain)
yTrain = np.array(yTrain)

# Define a custom dataset using PyTorch Dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(XTrain)
        self.x_data = XTrain
        self.y_data = yTrain

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Set hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(XTrain[0])
learning_rate = 0.001
num_epochs = 1000

# Create a dataset and dataloader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Check if CUDA (GPU) is available and move the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeutralNet(input_size, hidden_size, output_size).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model and related data to a file
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. File saved to {FILE}')
