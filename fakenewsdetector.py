# from flask import Flask, render_template, request
# import joblib
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# from sklearn.feature_extraction.text import TfidfVectorizer
# import nltk
# import os

# # Download NLTK resources
# nltk.download('stopwords')

# app = Flask(__name__)

# # Get the absolute path of the current directory
# base_dir = os.path.abspath(os.path.dirname(__file__))

# # Load the model and vectorizer
# model_path = os.path.join(base_dir, 'resources', 'models', 'model.sav')
# vectorizer_path = os.path.join(base_dir, 'resources', 'models', 'vectorizer.sav')

# fake_news_model = joblib.load(model_path)
# fake_news_vectorizer = joblib.load(vectorizer_path)

# # Function for text preprocessing
# def stemming(content):
#     port_stem = PorterStemmer()
#     stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
#     stemmed_content = stemmed_content.lower()
#     stemmed_content = stemmed_content.split()
#     stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
#     stemmed_content = ' '.join(stemmed_content)
#     return stemmed_content

# # Function to detect fake news
# def detect_fake_news(news_text):
#     # Preprocess the input text
#     preprocessed_text = stemming(news_text)
#     # Vectorize the text
#     text_vectorized = fake_news_vectorizer.transform([preprocessed_text])
#     # Predict using the model
#     prediction = fake_news_model.predict(text_vectorized)[0]
#     print(prediction)
#     return "Fake News" if prediction ==1 else "Real News"

# @app.route('/')
# def index():
#     return render_template('fakenews_form.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         headline = request.form['headline']
#         result_message = detect_fake_news(headline)
#         return render_template('fakenews_result.html', headline=headline, result_message=result_message)

# if __name__ == '__main__':
#     app.run(debug=True)
