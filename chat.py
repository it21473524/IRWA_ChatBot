import random
import json
import torch

from model import NeutralNet  # Import your NeuralNet model from model.py
from nltk_utils import bag_of_words, tokenize

# Check if a CUDA-compatible GPU is available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the intent data from a JSON file
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load the trained model and related data from a file
FILE = "data.pth"
data = torch.load(FILE)

# Extract data from the loaded file
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Create an instance of the NeuralNet model and load the pre-trained model state
model = NeutralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Mr.Bong"
def get_response(msg):
    sentence = tokenize(msg)  # Tokenize the user's input
    X = bag_of_words(sentence, all_words)  # Create a bag of words representation
    X = X.reshape(1, X.shape[0])  # Reshape for input to the model
    X = torch.from_numpy(X).to(device)

    # Pass the input through the model to get a prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Check if the model's prediction has a high confidence level
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        # If the confidence is high, randomly select and print a response from the intents data
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
        # If confidence is low, indicate that the bot does not understand
    return (f"{bot_name}: I do not understand...")


    
