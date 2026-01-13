Smart Agriculture Management System
Integrated Crop Prediction, Advisory, Weather Intelligence, Market Platform, and Storage Monitoring


1. Introduction

The Smart Agriculture Management System is an integrated software solution designed to support data-driven agricultural decision-making. The system combines machine learning–based crop prediction with soil advisory tools, weather analytics, market interaction features, and storage monitoring components. It aims to enhance productivity, reduce risks, and provide a unified platform for smart farming operations.

This repository contains all modules required to run, test, and deploy the system.

2. Key Features



2.1 Crop Recommendation System

Predicts the most suitable crop using a trained machine learning model (crop_model.joblib).

Utilizes soil nutrients, environmental parameters, and user inputs.

Stores prediction history in a structured database.

2.2 Agricultural Advisory Module

Provides fertilizer recommendations for nutrient imbalance.

Offers corrective suggestions based on soil conditions.

Implemented in crop_advisory.py.

2.3 Weather Analytics

Includes a backend engine for weather interpretation (weather_backend.py).

Maintains historical weather data (weather_data.db).

A dedicated testing environment is provided (weather_test.py).

2.4 Market Platform for Farmers

Enables crop listings, pricing, and trade-related interactions.

Facilitates a simple marketplace aimed at connecting buyers and sellers.

Implemented in market_platform.py.

2.5 Storage Monitoring System

Monitors warehouse/storage conditions such as capacity, moisture, and temperature.

Supports early detection of spoilage risks.

Managed through storage_monitor.py.

2.6 Backend Integration

All modules are orchestrated through a central application runner (app.py and main.py).

Multiple SQLite databases support persistent storage.

3. Repository Structure
Smart-Agriculture-System/
│
├── README.md
├── req.txt                     # Dependency list
├── app.py                      # Main application entry point
├── main.py                     # Central workflow controller
│
├── crop/                       # (Optional module folder)
├── crop_model.joblib           # Trained ML model
├── crop_advisory.py            # Soil and fertilizer advisory system
├── crop_data.db                # Crop dataset database
│
├── smart_agri.db               # Main integrated database
├── smartcrop_history.db        # Historical records of predictions
│
├── weather_backend.py          # Weather analysis backend
├── weather_data.db             # Weather logs database
├── weather_test.py             # Weather system test module
│
├── storage_monitor.py          # Storage/warehouse monitoring module
└── market_platform.py          # Marketplace module

4. Technology Stack
Programming Language

Python 3.8+

Libraries and Tools

Scikit-learn

NumPy

Pandas

SQLite3

Joblib

Standard Python utility modules

Data Storage

SQLite databases (*.db files)

5. Installation and Setup
Step 1: Install Dependencies
pip install -r req.txt

Step 2: Run the Application
python app.py

Step 3: Run Individual Modules (Optional)

Crop Prediction:

python main.py


Weather System:

python weather_test.py


Market Platform:

python market_platform.py


Storage Monitoring:

python storage_monitor.py

6. Module Descriptions
6.1 Crop Prediction

Accepts soil and environmental parameters as input.

Produces a recommended crop.

Uses a trained ML model stored in crop_model.joblib.

6.2 Advisory Module

Evaluates nutrient imbalances.

Recommends fertilizer application and corrective measures.

6.3 Weather System

Interprets weather trends and supports agricultural decisions.

Stores weather history for future analysis.

6.4 Market Platform

Allows creation and management of market listings.

Serves as a digital marketplace for farmers.

6.5 Storage Monitoring

Monitors physical conditions of agricultural storage units.

Ensures product quality and reduces spoilage risk.

7. Future Improvements

Integration with real-time weather APIs.

Mobile application support.

IoT sensor integration for soil and storage monitoring.

Advanced marketplace recommendation engine.

Dashboard development using Streamlit or a web framework.

8. Author

Sarthak Dhumal
