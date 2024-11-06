#importing the required libraries and modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
<<<<<<< Updated upstream
=======
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
>>>>>>> Stashed changes


#importing the dataset
data = pd.read_csv("c:/Users/bharg/Downloads/MSITM_Python/Professor_Git_Hub/NVIDIA-Stock-Analysis-Python/nvidia_stock_data.csv")

#converting the dataset into a dataframe
df = pd.DataFrame(data)
print(df.head(2))

#checking for null values in the dataset
null_count = df.isnull().sum()
print(null_count)

#exploring the dataset
print(df.describe())
print(df.count())

#checking the relation between the variables
df_1 = df.drop(columns='Date') #1. preprocessing step(Removing data to avoid errors as date is string and the rest are numeric)
correlation_matrix = df_1.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".5f")
plt.title("Correlation Matrix")
plt.savefig('correlation_matrix.jpg')
plt.show()
<<<<<<< Updated upstream
df_2 = df_1.drop(columns='Adj Close') #2. preprocessing step (Removing Adj Close to avoid multicollinearity)
=======
df_2 = df_1.drop(columns='Adj Close') #2. preprocessing step (Removing Adj Close to avoid multicollinearity)

#building a Linear Regression model (#1.machine learning model) to predict the close values with a train data of 80% and a test data of 20%
X = df_2[['Open', 'High', 'Low', 'Volume']] #independent variables
y = df_2['Close']#dependent variables
#splitting the dataset
X_train = X.iloc[:350]
y_train = y.iloc[:350]
X_test = X.iloc[350:437]
y_test = y.iloc[350:437]
#creating a model and training it
model = LinearRegression()
model.fit(X_train, y_train)

#making predictions based on the test set
y_linear_predictions = model.predict(X_test)

#model evaluation metrics for the Linear Regression Model
r2_linear = r2_score(y_test,y_linear_predictions)
mse_linear = mean_squared_error(y_test, y_linear_predictions)
mape_linear = mean_absolute_percentage_error(y_test, y_linear_predictions)
print(f"Mean Squared Error of the linear Regression model: {mse_linear:.4f}%")
print(f"R-squared of the linear Regression model: {r2_linear:.4f}%")
print(f"Mean Absolute Percentage Error of the linear Regression model: {mape_linear * 100:.2f}%")

#forecasting close value using ARIMA model(#2. machine Learning model)
close_series = y_train
adf_result = adfuller(close_series)
 
#fitting the ARIMA model
model = ARIMA(close_series, order=(1, 1, 1))
arima_result = model.fit()
 
#Summarizing the model
print(arima_result.summary())
#foreacsting the next 87 observations
forecast = arima_result.forecast(steps=87)
forecast_1 = arima_result.get_forecast(steps=87)
forecast_mean = forecast_1.predicted_mean
confidence_intervals = forecast_1.conf_int()
 
#testing accuracy
mse_arima = mean_squared_error(y_test, forecast)
mape_arima = mean_absolute_percentage_error(y_test, forecast)
print(f"Mean Squared Error of the ARIMA model: {mse_arima:.4f}")
print(f"Mean Absolute Percentage Error of ARIMA model: {mape_arima * 100:.2f}%")

# Plot the forecast along with historical data
plt.figure(figsize=(12, 6))
plt.plot(close_series, label='Historical Close')
plt.plot(forecast_mean, label='Forecast', color='orange')
plt.fill_between(confidence_intervals.index,
                 confidence_intervals.iloc[:, 0],
                 confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
plt.title("ARIMA Model Forecast for Close Values")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.savefig('ARIMA_Forecast.jpg')
plt.show()
 
#Comparing the results from ARIMA and Linear Regression
plt.plot(y_train, label='Train Data', color='blue')
plt.plot(y_test,label='Actual Data', color='green', linestyle='--', linewidth=2)
plt.plot(y_test.index,y_linear_predictions, label='Regression Forecast', color='orange', linestyle='dotted')
plt.plot(forecast_mean, label='ARIMA Forecast', color='red', linestyle='dotted')
 
#Add labels and legend to the plot
plt.title('Comparison of Regression and ARIMA Forecasts with Actual Data')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.savefig('Regression_vs_ARIMA.jpg')
plt.show()
>>>>>>> Stashed changes
