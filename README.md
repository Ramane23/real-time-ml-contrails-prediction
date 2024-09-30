**Real-Time ML Contrails (condensation trails) Prediction System**

**Project Overview**
This project aims to predict contrail formation in real-time for commercial flights using machine learning models and environmental data. Contrails are the condensation trails left behind by aircraft that can contribute significantly to global warming. By accurately predicting when and where contrails form, this project helps airlines reduce their environmental impact through optimized flight routes and operational strategies.

**Key Features**

**Data Integration:** Combines real-time flight tracking data from the Aviation Edge API and high-resolution weather data from the Meteomatics API.

**Feature Engineering:** Uses the OpenAP aircraft performance model to generate key aircraft performance features and the CoCiP (Contrail Cirrus Prediction) model from the PyContrails library to compute contrail formation features.

**Real-Time Inference:** Predicts contrail formation probabilities for ongoing flights, providing actionable insights for minimizing contrail formation.

**Deployment:** The project is fully containerized using Docker and deployed on Quix Cloud, with real-time data streaming managed through Redpanda.

**Experiment Tracking:** Uses Comet ML to monitor and track different model versions, hyperparameters, and performance metrics.
Architecture
The system is built using a modular microservices approach with the following components:

**Data Producers:** Extract live flight data and weather variables using APIs.

**Feature Engineering Module**: Processes and enriches flight data with meteorological and aircraft performance features.

**Model Inference Service:** Applies machine learning models to predict contrail formation and persistence.

**Visualization Dashboard:** Displays real-time predictions and historical contrail data using Streamlit and Plotly.

**API Integration:** FASTAPI-based REST API and WebSocket for serving real-time predictions and alerts.
Technical Stack

**Programming Languages:** Python, SQL

**Libraries:** LightGBM, PyContrails, Pandas, NumPy

**Data Storage:** Hopsworks Feature Store

**Deployment:** Docker, Docker-compose , Quix Cloud

**Data Streaming:** Redpanda (Message Queue)

**Experiment Tracking:** Comet ML

**APIs:** FastAPI (REST and WebSocket endpoints)

**Visualization:** Streamlit, Plotly

**Setup and Installation**

  # Clone the repository
  git clone https://github.com/Ramane23/real-time-ml-contrails-prediction.git

  # Change to the project directory
  cd real-time-ml-contrails-prediction

  # Build and Run Docker Containers:
  docker-compose up --build
  
**Configure Environment Variables:**
Set up your API keys and other environment variables in the .sh file as needed.

**Launch the Feature Dashboard: **

# Run the Streamlit dashboard:
streamlit run services/features_dashboard/src/frontend.py

**How to Contribute ?**

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make your changes and commit (git commit -m "Add new feature").
Push to your branch (git push origin feature-branch).
Open a Pull Request.

**Contact**

For more information or questions, feel free to reach out via contactsouley@gmail.com.
