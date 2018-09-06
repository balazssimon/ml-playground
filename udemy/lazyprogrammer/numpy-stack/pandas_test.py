import pandas as pd

X = pd.read_csv('data_2d.csv', header=None)

type(X)

X.info()

X.head()
X.head(10)

M = X.values   # as numpy array
type(M)

M[0] # 0th row of the matrix
X[0] # 0th column of the dataframe

type(X[0])

X.iloc[0] # 0th row of the dataframe

X[[0,2]]  # 0th and 2nd column

X[ X[0] < 5 ] # select all rows where the value in the 0th column is less than 5

X[0] < 5 # boolean series for the condition


df = pd.read_csv('international-airline-passengers.csv', engine='python', skipfooter=3)

df.columns # column names
df.columns = ["month", "passangers"] # reassigning column names
df["passangers"] # get a column
df.passangers # get a column

df["ones"] = 1 # add a column of numbers 1
df.head()

from datetime import datetime

datetime.strptime("1949-05","%Y-%m")
df['dt'] = df.apply(lambda row: datetime.strptime(row['month'],"%Y-%m"), axis=1)
df.info()
df.head()


t1 = pd.read_csv("table1.csv")
t2 = pd.read_csv("table2.csv")

m = pd.merge(t1, t2, on='user_id')
t1.merge(t2, on='user_id')

