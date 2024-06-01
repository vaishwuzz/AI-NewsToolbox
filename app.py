from flask import Flask, render_template, request
import joblib
import os
from summarizer import summarize_article  # Import your summarizer function
from transformers import pipeline
# from fake_news_detection import detect_fake_news  # Import the detect_fake_news function
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Initialize Flask app
app = Flask(__name__, static_folder='assets')

# Get the absolute path to the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the headline generation pipeline
headline_generator = pipeline("text-generation", model="gpt2")

# Function to preprocess text
def preprocess_text(text):
    # Add your text preprocessing logic if needed
    return text

# Function to generate headline
def generate_headline(news_text):
    # Generate headline using GPT-2 model with a shorter max_new_tokens value
    headline = headline_generator(news_text, max_new_tokens=20, num_return_sequences=1)[0]['generated_text']
    # Extract headline from response
    headline = headline.replace(news_text, '').strip()
    return headline

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/news_summarizer', methods=['POST', 'GET'])
def news_summarizer():
    if request.method == 'POST':
        article_text = request.form['article']
        num_sentences_in_summary = int(request.form['num_sentences'])
        summarized_text = summarize_article(article_text, num_sentences_in_summary)
        return render_template('news_summarizer.html', article=article_text, summary=summarized_text)
    return render_template('news_summarizer.html')



@app.route('/headline_generation', methods=['POST', 'GET'])
def headline_generation():
    if request.method == 'POST':
        news_text=request.form['news']
        generated_headline = generate_headline(news_text)
        
        return render_template('headline_generation.html', headline=generated_headline,news=news_text)
    return render_template('headline_generation.html')

tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    tfid_x_train = tfvect.fit_transform(x_train)
    tfid_x_test = tfvect.transform(x_test)
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

# @app.route('/fake_news_detector')
# def home():
#     return render_template('index.html')

# @app.route('/fake_news_detector', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # message = request.form['message']
#         article_text = request.form['article']
#         pred = fake_news_det(article_text)
#         print(pred)
#         return render_template('fake_news_detector.html',article=article_text, result=pred)
#     else:
#         return render_template('fake_news_detector.html')
@app.route('/fake_news_detector', methods=['POST', 'GET'])
def fake_news_detector():
    if request.method == 'POST':
        article_text = request.form['article']
        result = fake_news_det(article_text)  # Use the imported function
        print()
        return render_template('fake_news_detector.html', article=article_text, result=result)
    return render_template('fake_news_detector.html')
    
if __name__ == '__main__':
    app.run(debug=True)
