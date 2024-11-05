import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

with open("vectorizer.pkl","rb") as file:
    vectorizer = pickle.load(file)


class TextClassifier(nn.Module):
  def __init__(self,input_size):
    super(TextClassifier,self).__init__()
    self.fc1 = nn.Linear(input_size,128)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,6)

  def forward(self,x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    return x
  

model = TextClassifier(5000)
model.load_state_dict(torch.load('model.pth'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

stops =stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocessText(text):
  tokens = word_tokenize(text.lower())
  tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha() and token not in stops]
  return ' '.join(tokens)

def predict(text):

    emotion = {0:'sad', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}

    text = preprocessText(text)
    text_vector = vectorizer.transform([text]).toarray()  # Make sure it's a list
    text_vector = torch.tensor(text_vector, dtype=torch.float32).to(device)  # Move tensor to the device
    
    model.to(device)  # Move model to the same device
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient tracking for inference
        output = model(text_vector)
    
    probabilities = F.softmax(output, dim=1)

    # Get the predicted class index
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return emotion[predicted_class]


st.title("Emotion Classifier")

text = st.text_area("Start writing....")

_,mid_col,_ = st.columns(3)

if mid_col.button("Classify"):
   st.write(predict(text))

