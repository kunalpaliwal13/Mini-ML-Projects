from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the trained model
with open('spam_classifier_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        input_data = [message]

        # Convert text to feature vectors
        input_data_features = vectorizer.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(input_data_features)

        if prediction == 0:
            result = "It's a spam mail."
        else:
            result = "It's a ham mail."

        return render_template('result.html', message=message, result=result)

if __name__ == '__main__':
    app.run(debug=True)
