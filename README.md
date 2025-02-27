# Titanic Survival Prediction

## Overview

This project focuses on predicting passenger survival on the Titanic based on various features using a machine learning technique. The dataset is sourced from the well-known Titanic dataset, which includes information such as age, gender, ticket class, and more. The goal is to apply machine learning algorithms to predict whether a passenger survived or not.

## Project Structure

- **Titanic\_Survival\_Prediction.ipynb**: The main Jupyter notebook containing the data exploration, feature engineering, model training, and evaluation for the Titanic survival prediction task.
- **app.py**: The Flask web application for deploying the model.
- **templates/**: Directory containing the HTML templates for the web application.
- **static/**: Directory for CSS and JavaScript files.
- **requirements.txt**: The list of dependencies required to run the project.
- **Data/**: The Titanic dataset used in this project, typically available from Kaggle or preloaded in the notebook.

## Requirements

To run this project locally, ensure you have the following installed:

- Python 3.x
- Jupyter Notebook
- Flask
- Gunicorn
- Necessary libraries such as:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

You can install these dependencies using the following command:

```sh
pip install -r requirements.txt
```

(Note: You will need to create a requirements.txt file based on the libraries used in your notebook. You can generate it using `pip freeze > requirements.txt`.)

## How to Run Locally

1. Clone this repository or download the files.
2. Ensure that the dataset is loaded and available within the notebook (the dataset should either be linked or included in the project folder).
3. Open the **Titanic\_Survival\_Prediction.ipynb** file in Jupyter Notebook and run all cells sequentially to execute the code.
4. To run the web application locally:
   ```sh
   python app.py
   ```
5. Open `http://127.0.0.1:5000/` in your browser to interact with the web app.

## Deployment on Render

This project has been deployed using **Flask** and **Render**. The live web application can be accessed at:

[Survival Prediction Web App](https://survival-prediction-using-titanic-dataset.onrender.com)

### Steps to Deploy on Render:

1. Create an account on [Render](https://render.com/).
2. Connect your GitHub repository to Render.
3. Create a new **Web Service** on Render.
4. Set the **Start Command** to:
   ```sh
   gunicorn titanicapp:app
   ```
5. Deploy the application and obtain the live URL.

## Data Description

The dataset contains the following key features:

- **PassengerId**: Unique ID for each passenger.
- **Pclass**: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- **Name**: Name of the passenger.
- **Sex**: Gender of the passenger.
- **Age**: Age of the passenger.
- **SibSp**: Number of siblings or spouses aboard.
- **Parch**: Number of parents or children aboard.
- **Ticket**: Ticket number.
- **Fare**: Fare paid for the ticket.
- **Cabin**: Cabin number (if available).
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).
- **Survived**: Target variable (0 = Did not survive, 1 = Survived).

## Analysis and Methodology

The notebook performs the following steps:

### Exploratory Data Analysis (EDA):

- Visualization of data distribution and relationships between features using tools like Seaborn and Matplotlib.
- Handling of missing data (e.g., missing age or cabin information).

### Feature Engineering:

- Transforming categorical features (e.g., converting "Sex" to numerical values).
- Creating new features or adjusting existing ones to improve model performance.

### Model Selection and Training:

- Application of Logistic Regression.
- Tuning hyperparameters and evaluating model performance using accuracy, precision, recall, and F1-score.

### Model Evaluation:

- Confusion matrix and other relevant metrics to assess model accuracy.

## Results

The model is evaluated based on its performance in predicting Titanic survival. The key findings and results of the best model will be highlighted in the notebook.

## Acknowledgements

- **Dataset**: Kaggle Titanic Dataset
- **Tools**: Python, Flask, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Gunicorn, Render

## License

This project is open-source and available under the MIT License.

