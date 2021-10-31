# Projeto: Machine Learning: Linguagens de Programacao 2004-2001

Projeto de Data Science e Machine Learning de análise de linguagens de programação de 2004 a 2021 obtidos a partir do seguinte dataset do [Kaggle](https://www.kaggle.com/muhammadkhalid/most-popular-programming-languages-since-2004).

### Tecnologias

* Python 3
* Jupyter Notebook
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Requests
* REST API Call (Github API)

### Algoritmos

* Regressão Linear

Inicialmente, serão visualizados dados de séries temporais e regressão linear.

### Tratamento de dados

```python
df = pd.read_csv('Most Popular Programming Languages from 2004 to 2021 V4.csv')

def createDataFrameFor(df, colunas, colunaAtual):
    return pd.DataFrame(
        {
            'Date': df.Date,
            'Timestamp': map(lambda i : datetime.strptime(df["Date"][i], '%B %Y'), range(len(df.Date))),
            'Language': colunas[colunaAtual],
            'Value': df[df.columns[colunaAtual]]
        }
    )

colunas = df.columns

dados_tratados = createDataFrameFor(df, colunas, 1)

for coluna in range(1, len(colunas)):
    dados_tratados = pd.concat([dados_tratados, createDataFrameFor(df, colunas, coluna)])

dados_tratados.reset_index(drop=True, inplace=True)

dados_tratados['UnixTime'] = list(map(lambda i: (pd.to_datetime([dados_tratados['Timestamp'][i]]).astype(int) / 10**9)[0], range(len(dados_tratados['Date']))))

```

### Visualização dos dados

```python
df_java = dados_tratados[dados_tratados['Language'] == 'Java']
sns.regplot(x="UnixTime", y="Value", data= df_java)
plt.gcf().set_size_inches(16, 6)
plt.ylabel('% de uso da linguagem Java')
plt.xlabel('Anos em Unix Time de 2004 a 2021')
plt.show()

X = df_java.UnixTime.values.reshape(-1, 1)
y = df_java.Value.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

plt.scatter(X_test, y_test,  color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=3)

plt.gcf().set_size_inches(16, 6)

plt.ylabel('% de uso da linguagem Java')
plt.xlabel('Anos em Unix Time de 2004 a 2021')

plt.show()
```

### Plots gerados

* Regressão Linear com Seaborn (SNS):

![Seaborn](https://github.com/vhnegrisoli/machine-learning-linguagens-programacao/blob/master/imgs/Figure_1.png)

* Regressão Linear com Scikit-Learn (LinearRegression) e Matplotlib:

![Matplotlib ScikitLearn](https://github.com/vhnegrisoli/machine-learning-linguagens-programacao/blob/master/imgs/Figure_2.png)

### Plots de evolução das linguagens Java, Javascript, Python, C#, PHP, Delphi, Dart e Cobol nos últimos 17 anos

```python
dados = df

dados['Date'] = pd.to_datetime(dados['Date'])
dados.set_index('Date', inplace = True) 

fig, axes = plt.subplots(nrows=4, ncols=2)

dados['Java'].plot(ax=axes[0,0], title = "Análise da linguagem Java nos últimos 17 anos")
dados['JavaScript'].plot(ax=axes[1,0], title = "Análise da linguagem JavaScript nos últimos 17 anos")
dados['Python'].plot(ax=axes[0,1], title = "Análise da linguagem Python nos últimos 17 anos")
dados['C#'].plot(ax=axes[1,1], title = "Análise da linguagem C# nos últimos 17 anos")
dados['PHP'].plot(ax=axes[2,0], title = "Análise da linguagem PHP nos últimos 17 anos")
dados['Delphi'].plot(ax=axes[2,1], title = "Análise da linguagem Delphi nos últimos 17 anos")
dados['Dart'].plot(ax=axes[3,0], title = "Análise da linguagem Dart nos últimos 17 anos")
dados['Cobol'].plot(ax=axes[3,1], title = "Análise da linguagem Cobol nos últimos 17 anos")

plt.gcf().set_size_inches(16, 22)

plt.show()
```

![17 anos](https://github.com/vhnegrisoli/machine-learning-linguagens-programacao/blob/master/imgs/Compara%C3%A7%C3%A3o%20linguagens%2017%20anos.png)

### Autor

* Victor Hugo Negrisoli
* Desenvolvedor Back-End Sênior | Analista de Dados
