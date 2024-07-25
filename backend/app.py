from flask import Flask, request, jsonify
from flask_cors import CORS
from distSystem import main

app = Flask(__name__)
CORS(app)

@app.route('/start-training', methods=['GET'])
def start_training():
    dataset = request.args.get('dataset', default='MNIST')
    epochs = int(request.args.get('epochs', default=10))
    if epochs < 1:
        return jsonify({"error": "Epochs must be at least 1"}), 400  # Bad Request
    
    results = perform_training(dataset, epochs)
    return jsonify(results)

def perform_training(dataset, epochs):
    if dataset == 'MNIST':
        time_dict = main(epochs=epochs)
        time_dict = dict(time_dict)
        return time_dict

if __name__ == "__main__":
    app.run(debug=True)
