#importing the required libraries and modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
df_2 = df_1.drop(columns='Adj Close') #2. preprocessing step (Removing Adj Close to avoid multicollinearity)