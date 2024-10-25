<div align="center">
    <h1>Real-Time ML Contrails (condensation trails) Prediction System
</div>

# Hello, I'm Souleymane 👋

## 🚀 About Me
I am a Machine Learning and Aeronautical Engineer passionate about building end-to-end data-driven solutions for real-time aviation and sustainability challenges. With experience in deploying production-grade models, I aim to leverage AI to create a sustainable impact.

- 🔭 I’m currently working on a real-time contrail prediction system.
- 🌱 I enjoy building End-to-End Machine Learning Systems (batch and real-time).
- 💬 Ask me about anything related to machine learning, MLOps, or data engineering.

# System Architecture:
![image](https://github.com/user-attachments/assets/af6b619c-0e2c-44de-a132-8fc07d9f10d0)

#### Table of contents
* [Project Overview ](#project-overview)
* [How to run the features_pipeline locally with docker-compose ](#how-to-run-the-features-pipeline-locally-with-docker-compose?)
* [How to run the streamlit dashboard ? ](#how-to-run-the-streamlit-dashboard?)
* [How to run the training script locally? ](#how-to-run-the-training-script-locally?)
* [How to run the FastAPI for inference locally? ](#how-to-run-the-FastAPI-for-inference-locally?)


## Project Overview
This project aims to predict contrail formation in real-time for commercial flights using machine learning models and environmental data. Contrails are the condensation trails left behind by aircraft that can contribute significantly to global warming. By accurately predicting when and where contrails form, this project helps airlines reduce their environmental impact through optimized flight routes and operational strategies.

## How to run the features_pipeline locally with docker-compose?

Git clone this repository, cd into the root directory of the projec, then in the docker-compose directory and then run the following commands using make.

1. Install [Python Poetry](https://python-poetry.org/docs/#installation) (if necessary)
and create an isolated virtual environmnet for development purposes.

2. Test, build and run the dockerized features pipeline with docker-compose
    ```
    $ make build-live-feature-pipeline
    $ make run-live-feature-pipeline
    
    ```
## How to run the streamlit dashboard?
cd into the features_dahboard service and run:

    ```
    $ make run-dev
    ```
## How to run the training script locally?
cd into the contrail_predictor service and run:
    ```
    $ make train
    ``` 
## How to run the FastAPI and make inference locally?
cd into the contrail_predictor service and run:
    ```
    $ make restapi
    $ make check-health
    $ make predict
    ``` 



**Contact**

For more information or questions, feel free to reach out via contactsouley@gmail.com.
