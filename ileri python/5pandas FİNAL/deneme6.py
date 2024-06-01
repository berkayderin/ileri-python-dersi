import pandas as pd

df1=pd.read_csv('googleplaystore.csv')
print(df1.shape)
print(type(df1))
print(df1.dtypes)
print('-'*50)
print(df1.columns)
print('-'*50)
movies_df=pd.read_csv('IMDB-Movie-Data.csv',index_col='Title')
print(movies_df.shape)
print(movies_df.columns)
print('-'*50)

print(df1.head())
print(df1.tail(2)[['Category','Rating']])

print(df1.loc[:,'Size':'Genres'])
print('-'*50)

print(df1.info())
print(movies_df.info())
print('-'*50)

print(df1.isnull().sum().sort_values(ascending=False))

print(df1.describe())
print(df1['Category'].describe())
print(df1.describe(include=['O']))
print(df1['Category'].value_counts())

print(df1[df1['Category']=='1.9'])
df1._set_value(10472,'Category','PHOTOGRAPHY')
print(df1['Category'].value_counts())
print(df1.sort_values(by='Rating',ascending=False).head(10)[['App','Rating']])

df1.drop(10472,inplace=True)
print(df1.sort_values(by='Rating',ascending=False).head(10)[['App','Rating']])
df1.drop('Content Rating',axis=1,inplace=True)
print(df1.columns)

print(df1.groupby(['Category','Type'])['Rating'].mean().head(10))
print(df1.info())
df1['Rating'].fillna(df1.groupby('Category')['Rating'].transform('mean'),inplace=True)
print(df1.info())
print(df1.isnull().sum().sort_values(ascending=False))
df1.dropna(inplace=True)
print(df1.isnull().sum().sort_values(ascending=False))
df2=df1.groupby(['Category','Type'])['Rating'].mean().head(10)
print(df2)
df2.to_excel('aaa.xlsx')
