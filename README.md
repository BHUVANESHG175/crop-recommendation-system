\# Smart Crop Recommender



\## About the Project



Smart Crop Recommender is a web application I built using Django and machine learning. The main goal of this project is to recommend a suitable crop based on soil nutrients and environmental conditions.



The user enters values such as nitrogen, phosphorus, potassium, temperature, humidity, soil pH, and rainfall. The application uses a trained Random Forest model to predict which crop may be suitable for those conditions.



I also added user accounts, prediction history, and an admin dashboard to make the application more useful.



\## Features



\- \*\*Crop prediction:\*\* Recommends a crop based on the entered values.

\- \*\*Input validation:\*\* Checks the entered values and rejects invalid inputs.

\- \*\*User accounts:\*\* Allows users to create an account, log in, and manage their profile.

\- \*\*Prediction history:\*\* Lets users view their previous crop predictions.

\- \*\*Weather features:\*\* Includes functionality for retrieving climate information.

\- \*\*Admin dashboard:\*\* Allows administrators to view application statistics and manage users.

\- \*\*Automated tests:\*\* Includes tests for important application features.

\- \*\*Docker setup:\*\* Includes files for running the application in a Docker container.



\## Technologies Used



\- Python

\- Django

\- scikit-learn

\- pandas and NumPy

\- HTML, CSS, and Bootstrap

\- SQLite

\- pytest and Playwright

\- Docker and Gunicorn



\## How the Project Works



1\. The user logs in to the application.

2\. The user enters soil and environmental values.

3\. The application checks whether the inputs are valid.

4\. The trained Random Forest model predicts a crop.

5\. The prediction is displayed to the user and saved in the prediction history.



The model uses seven input features:



\- Nitrogen (N)

\- Phosphorus (P)

\- Potassium (K)

\- Temperature

\- Humidity

\- Soil pH

\- Rainfall



The recommendation depends on the training data and the accuracy of the model. It should be used as a starting point, not as a guarantee that a crop will grow successfully.



\## Getting Started



Follow these steps to run the project on your computer.



\### 1. Requirements



You will need:



\- Python 3.12

\- pip

\- A terminal or Command Prompt



Docker is optional if you want to try running the application in a container.



\### 2. Open the project folder



Open Command Prompt and move to the project directory:



```bat

cd C:\\crop\_recommendation\_site

```



\### 3. Activate the virtual environment



If you already created the virtual environment, activate it:



```bat

.venv\\Scripts\\activate

```



If you have not created it yet, run:



```bat

python -m venv .venv

.venv\\Scripts\\activate

```



\### 4. Install the required packages



Run:



```bat

python -m pip install -r requirements.txt

```



This installs the Python packages listed in `requirements.txt`.



\### 5. Set up the database



Run the database migrations:



```bat

python manage.py migrate

```



If you need to create an administrator account, run:



```bat

python manage.py createsuperuser

```



Follow the instructions shown in the terminal.



\### 6. Start the application



Run:



```bat

python manage.py runserver 8080

```



Open this address in your browser:



http://127.0.0.1:8080/



To stop the server, press `Ctrl+C` in Command Prompt.



\## Running the Tests



I included tests to check important parts of the application.



To run the Django tests:



```bat

python manage.py test recommender

```



To run the pytest tests:



```bat

python -m pytest -q

```



To check the Django project for configuration problems:



```bat

python manage.py check

```



These commands can help identify problems when the application is changed or updated.



\## Running with Docker



The project includes a `Dockerfile` and a `.dockerignore` file.



If Docker Desktop is installed and running, build the image from the project directory:



```bat

docker build -t smart-crop-recommender .

```



After the image builds successfully, run:



```bat

docker run --rm -p 8000:8000 smart-crop-recommender

```



Then open:



http://127.0.0.1:8000/



The Docker setup still needs to be tested on the machine where you intend to run it. For a public deployment, database persistence, secret settings, allowed hosts, and static files also need to be configured correctly.



\## Project Structure



Here are some of the main files and folders:



```text

crop\_recommendation\_site/

├── crop\_recommendation\_site/

│   ├── settings.py

│   ├── urls.py

│   └── wsgi.py

├── recommender/

│   ├── ml.py

│   ├── weather.py

│   ├── views.py

│   ├── urls.py

│   ├── models.py

│   ├── tests.py

│   └── templates/

├── Dockerfile

├── .dockerignore

├── requirements.txt

├── manage.py

└── README.md

```



\## What I Learned



While working on this project, I gained practical experience with:



\- Building a web application using Django.

\- Connecting a machine learning model to a web application.

\- Validating user input before making predictions.

\- Working with a database to store prediction history.

\- Adding user authentication and admin features.

\- Writing automated tests and debugging errors.

\- Using Git to track changes in the project.

\- Preparing a project for Docker.



\## Future Improvements



Some improvements I would like to work on next are:



\- Measuring the model's accuracy with a separate test dataset.

\- Improving the model evaluation process.

\- Tracking model versions with MLflow.

\- Deploying the application with secure settings and a persistent database.

\- Monitoring prediction inputs and model performance.



\## Note



This project is for learning and demonstration purposes. Crop recommendations depend on the training data and the conditions represented by the model. Actual farming decisions should also consider local soil tests, climate, crop varieties, and agricultural guidance.

