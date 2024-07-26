function start_training() {
    const dataset = document.getElementById('dataset').value;
    let epochs = parseInt(document.getElementById('epochs').value, 10);
    let learning_rate1 = parseFloat(document.getElementById('learning_rate1').value);
    let learning_rate2 = parseFloat(document.getElementById('learning_rate2').value);

    if (epochs < 1 || epochs > 10) {
        alert("Epochs must be at least 1 and at most 10.");
        return;  // Stop the function if validation fails
    }

    if (learning_rate1 < 0.01 || learning_rate2 < 0.01) {
        alert("Learning rate must be at least 0.01.")
        return
    }

    // Hide the button
    document.getElementById('startButton').style.display = 'none';

    fetch(`http://localhost:5000/start-training?dataset=${dataset}&epochs=${epochs}&lrone=${learning_rate1}&lrtwo=${learning_rate2}`, { method: 'GET' })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('results').innerHTML = data.message;
        waitForCompletion();
    })
    .catch(error => {
        console.error('Error starting training:', error)
        // Show the button in case of error to allow retrying
        document.getElementById('startButton').style.display = 'block';
    });
}

function waitForCompletion() {
    fetch('http://localhost:5000/check-complete')
        .then(response => response.json())
        .then(data => {
            if (data.isComplete) {
                document.getElementById('results').innerHTML = `<b>Distributed model training (world_size=3)</b> <br> Time: ${data.results.Distributed_training} <br> F1-score: ${data.results.f1_score1} <br> Accuracy: ${data.results.accuracy1} <br> Recall: ${data.results.recall1} <br> Precision: ${data.results.precision1} <br> <br> <br> <b>Non-distributed model training</b> <br> Time: ${data.results.Non_distributed_training} <br> F1-score: ${data.results.f1_score2} <br> Accuracy: ${data.results.accuracy2} <br> Recall: ${data.results.recall2} <br> Precision: ${data.results.precision2}`;
                // Show the button again once training is complete
                document.getElementById('startButton').style.display = 'block';
            } else {
                setTimeout(waitForCompletion, 2000); // Poll every 2 seconds
            }
        });
}