from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load pickel file for vectorizer
with open("tfidf.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

# Load pickel file for model
with open("svm_best_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

# Function to convert prediction to label
def sentimen_label(prediction):
    if prediction == 'Negatif':
        return 'Negatif'
    elif prediction == 'Positif':
        return 'Positif'

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        vector = vectorizer.transform([text])
        predicted_sentimen = model.predict(vector)[0]
        predicted_sentimen_label = sentimen_label(predicted_sentimen)
        return render_template('index.html', text=text, prediksi=predicted_sentimen_label)
    return render_template('index.html')

if __name__ == '__main__':
    app.run()