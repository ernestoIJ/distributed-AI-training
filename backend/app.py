from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread
from distSystem import main

app = Flask(__name__)
CORS(app)

is_training_complete = False
results = {}

@app.route('/start-training', methods=['GET'])
def start_training():
    global is_training_complete
    is_training_complete = False
    dataset = request.args.get('dataset', default='MNIST')
    epochs = int(request.args.get('epochs', default=2))
    lr1 = float(request.args.get('lrone', default=0.01))
    lr2 = float(request.args.get('lrone', default=0.01))

    if epochs < 1 or epochs > 10:
        return jsonify({"error": "Epochs must be at least 1 and at most 10"}), 400  # Bad Request
    
    if lr1 < 0.01 or lr2 < 0.01:
        return jsonify({"error": "Learning rate must be at least 0.01"}), 400  # Bad Request

    thread = Thread(target=perform_training, args=(dataset, epochs, lr1, lr2))
    thread.start()
    return jsonify({"message": "Training both models (distributed and non-distributed)..."})

@app.route('/check-complete', methods=['GET'])
def check_complete():
    return jsonify({"isComplete": is_training_complete, "results": results})

def perform_training(dataset, epochs, lr_1, lr_2):
    global is_training_complete
    global results
    if dataset == 'MNIST':
        time_dict = main(epochs=epochs, lr_1=lr_1, lr_2=lr_2)
        time_dict = dict(time_dict)
        is_training_complete = True
        results = time_dict

if __name__ == "__main__":
    app.run(debug=True)