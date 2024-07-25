function start_training() {
    const dataset = document.getElementById('dataset').value;
    let epochs = parseInt(document.getElementById('epochs').value, 10);

    if (epochs < 1 || epochs > 10) {
        alert("Epochs must be at least 1 and at most 10.");
        return;  // Stop the function if validation fails
    }

    fetch(`http://localhost:5000/start-training?dataset=${dataset}&epochs=${epochs}`, { method: 'GET' })
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
    .catch(error => console.error('Error starting training:', error));
}

function waitForCompletion() {
    fetch('http://localhost:5000/check-complete')
        .then(response => response.json())
        .then(data => {
            if (data.isComplete) {
                document.getElementById('results').innerHTML = `<b>Distributed model training</b> <br> Time: ${data.results.Distributed_training} <br> F1-score: ${data.results.f1_score1} <br> Accuracy: ${data.results.accuracy1} <br> Recall: ${data.results.recall1} <br> Precision: ${data.results.precision1} <br> <br> <br> <b>Non-distributed model training</b> <br> Time: ${data.results.Non_distributed_training} <br> F1-score: ${data.results.f1_score2} <br> Accuracy: ${data.results.accuracy2} <br> Recall: ${data.results.recall2} <br> Precision: ${data.results.precision2}`;
            } else {
                setTimeout(waitForCompletion, 2000); // Poll every 2 seconds
            }
        });
}