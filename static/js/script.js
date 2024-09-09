document.getElementById('upload-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const formData = new FormData();
    const imageFile = document.getElementById('imageUpload').files[0];
    formData.append('image', imageFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('knn-result').innerText = data.knn_prediction;
            document.getElementById('svm-result').innerText = data.svm_prediction;
            document.getElementById('cnn-result').innerText = data.cnn_prediction;
            document.getElementById('ensemble-result').innerText = data.ensemble_prediction;
        } else {
            alert('Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again.');
    }
});
