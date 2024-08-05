COCOA MARKET PRICE PREDICTOR

Youtube demo link: https://youtu.be/0ZXm7gTz_BA?feature=shared

This machine learning project is a cocoa price prediction app that uses Streamlit. It uses LSTM and GRU models to predict future prices based on previous data (Ghana Stock Exchange). The application allows for input of a time window and a graphical view of the predicted prices along with actual prices for comparison.

Features
- Downloads LSTM and GRU models and a scaler from Google Drive
- Loading stock price data
- Uses the loaded models to predict stock prices for a specified date range.
- Displays a graph plot comparing actual and predicted prices.

Requirements
- Python 3.7+
- Streamlit
- pandas
- numpy
- joblib
- gdown
- matplotlib

Installation
- clone the repository
- install the required packages
- download the pre-trained models
- place the stock prices data CSV file (CFC_traded_shares_2019_to_date.csv) in the project directory

Usage
- run the Streamlit app
- input the desired time range for prediction and view the results.

Local server hosting
Ensure you have followed the installation steps above. Run the Streamlit app using the following command: streamlit run app2.py

Hosting on the Cloud
- Create a Procfile
- Create a requirements.txt file
- Deploy to Heroku:
	- heroku login
	- heroku create
	- git add .
	  git commit -m "Initial commit"
	  git push heroku master
	- heroku open


	