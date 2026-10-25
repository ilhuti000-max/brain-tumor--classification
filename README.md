# Brain Tumor Classification AI


A deep learning research project that classifies brain MRI images into two categories: Tumor and No Tumor. The project uses a trained Keras model and a Streamlit web interface to make image-based predictions.

Features

- Upload a brain MRI image.
- Preview the uploaded image.
- Predict the class: Tumor or No Tumor.
- Display prediction confidence.
- Simple, user-friendly healthcare dashboard.

Technology Stack

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

Project Structure

brain-tumor--classification/

├── app.py

├── brain_tumor_model.keras

├── requirements.txt

└── README.md

Installation

1. Clone the repository

git clone https://github.com/ilhuti000-max/brain-tumor--classification.git

cd brain-tumor--classification

2. Create a virtual environment

python -m venv .venv

Activate it on Windows:

.\.venv\Scripts\Activate.ps1

3. Install dependencies

pip install -r requirements.txt

4. Run the application

streamlit run app.py

Open the local URL displayed in your terminal.


 [Open the Live Dashboard](https://brain-tumor--classification-6dbhv8yyhmhcgudezxwbb7.streamlit.app/)

An AI-powered brain tumor image classification project built with TensorFlow and Streamlit.


Model Evaluation

In the current held-out test results, the model achieved approximately 99% accuracy across 1,600 images.

The confusion matrix showed:

- 400 correctly classified no-tumor images.
- 1,191 correctly classified tumor images.
- 9 tumor images incorrectly classified as no tumor.
- 0 no-tumor images incorrectly classified as tumor.

These results are preliminary and depend on the dataset and evaluation procedure. They do not establish clinical performance or guarantee reliable predictions on new patients.

Limitations and Disclaimer

This project is intended for educational and research purposes only. It is not a medical device and must not be used to diagnose, rule out, or make treatment decisions about brain tumors. Model predictions may be incorrect. MRI scans should be interpreted by qualified medical professionals.

Future Improvements

- Investigate and reduce false-negative tumor predictions.
- Evaluate performance on independent, patient-level test data.
- Assess sensitivity, specificity, precision, and calibration.
- Improve generalization to MRI images from different sources.
- Enhance the user interface and testing process.

License

Add a license only after confirming that you have the rights to distribute the source code, trained model, and any included assets.
