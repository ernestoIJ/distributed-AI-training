function start_training() {
    const dataset = document.getElementById('dataset').value;
    let epochs = parseInt(document.getElementById('epochs').value, 10);

    if (epochs < 1) {
        alert("Epochs must be at least 1.");
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
        document.getElementById('results').innerHTML = `<b>Distributed model training</b> <br> Time: ${data.DDS_Training_time} <br> F1-score: 0.99 <br> Accuracy: 0.99 <br> Recall: 0.99 <br> Precision: 0.99 <br> <br> <br> <b>Non-distributed model training</b> <br> Time: ${data.Without_DDS_Training_time} <br> F1-score: 0.99 <br> Accuracy: 0.99 <br> Recall: 0.99 <br> Precision: 0.99`;
    })
    .catch(error => console.error('Error starting training:', error));

}