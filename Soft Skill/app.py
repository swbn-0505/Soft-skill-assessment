from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize the Flask application
app = Flask(__name__)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Result page route
@app.route('/result', methods=['POST'])
def result():
    # Get user input from the form
    communication = int(request.form['communication'])
    teamwork = int(request.form['teamwork'])
    problem_solving = int(request.form['problem_solving'])
    leadership = int(request.form['leadership'])
    adaptability = int(request.form['adaptability'])
    emotional_intelligence = int(request.form['emotional_intelligence'])

    # Create an input array for prediction
    user_input = np.array([[communication , teamwork , problem_solving , leadership , adaptability , emotional_intelligence]])

    # Predict the career based on user input
    predicted_result = model.predict(user_input)[0]

    return render_template('result.html', career=predicted_result)

if __name__ == '__main__':
    app.run(debug=True, port=5500)
