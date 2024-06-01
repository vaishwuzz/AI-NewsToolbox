import re
from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Initialize headline generation pipeline
headline_generator = pipeline("text-generation", model="gpt2")

# Function to remove special characters from a string
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

# Function to generate a cleaned headline
def generate_headline(news_text):
    # Generate headline using GPT-2 model
    generated_headline = headline_generator(news_text, max_length=100, num_return_sequences=1)[0]['generated_text']

    # Clean the generated headline
    cleaned_headline = remove_special_characters(generated_headline)

    # Post-processing
    cleaned_headline = cleaned_headline.strip()  # Remove leading and trailing spaces
    if cleaned_headline and not cleaned_headline[0].isalpha():  # Check if the first character is not a letter
        cleaned_headline = cleaned_headline.lstrip('.').strip()  # Remove leading '.' and spaces

    return cleaned_headline.capitalize()  # Capitalize the first letter

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_headline', methods=['POST'])
def generate_headline_route():
    if request.method == 'POST':
        # Get news text from form
        news_text = request.form['news']

        # Generate cleaned headline
        cleaned_headline = generate_headline(news_text)

        return render_template('index.html', news=news_text, headline=cleaned_headline)
    else:
        return "Method Not Allowed", 405

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

if __name__ == '__main__':
    app.run(debug=True)
