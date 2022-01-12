import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import svm

import warnings
warnings.filterwarnings('ignore')

def configurar_plot_com_dimensoes(titulo, x, y, h, w):
    plt.title(titulo)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.gcf().set_size_inches(h, w)
    plt.show()

def configurar_plot(titulo, x, y):
    configurar_plot_com_dimensoes(titulo, x, y, 16, 8)

df = pd.read_csv('dados/Most Popular Programming Languages from 2004 to 2021 V4.csv')
df.head(10)

df.shape

df.info()

df.describe()

meses = sorted(list(set([i.split(' ')[0] for i in df['Date'].unique()])))
anos = sorted(list(set([i.split(' ')[1] for i in df['Date'].unique()])))
print('Meses: {}\nAnos: {}'.format(meses, anos))

for column in df.columns:
    print(column)
print('\nTotal de colunas: {}.'.format(len(df.columns)))

df.hist()
configurar_plot_com_dimensoes('Histograma Linguagens', '', '', 20, 20)
plt.show()

def createDataFrameFor(df, colunas, colunaAtual):
    return pd.DataFrame(
        {
            'Date': df.Date,
            'Year': pd.DatetimeIndex(df['Date']).year,
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

dados_tratados['UnixTime'] = list(map(lambda i:                                    (pd.to_datetime([dados_tratados['Timestamp'][i]]).astype(int) / 10**9)[0],                                   range(len(dados_tratados['Date']))))

dados_tratados.head()

dados_tratados.to_csv('dados/Dados.csv')

dados_tratados.info()

dados_tratados.shape

dados_tratados.describe()

dados_agrupados = dados_tratados[['Language', 'Value']].groupby(by=['Language'], as_index=False).sum()

dados_agrupados = dados_agrupados.sort_values(by=['Value'], ascending=False)

dados_agrupados

fig, ax = plt.subplots()

ax.set_xticklabels(dados_agrupados['Language'])

ax.bar(x = dados_agrupados['Language'], height = dados_agrupados['Value'])
plt.gcf().set_size_inches(35, 10)
plt.savefig('imgs/Analise Linguagens.png')

configurar_plot_com_dimensoes(
    'Análise de linguagens mais usadas entre 2004 e 2021.',
    'Linguagens de programação.',
    '% de uso.',
    35, 
    10
)

# ### Verificando correlação entre as variáveis

# * Igual a 1 ---------> Correlação linear positiva perfeita
# * Maior que 0 -----> Correlação linear positiva
# * Igual a 0 ---------> Sem correlação linear
# * Menor que 0 ----> Correlação linear negativa
# * Igual a -1 --------> Correlação linear negativa perfeita

targets = [
    {
        'linguagem_1': 'Java',
        'linguagem_2': 'JavaScript'
    },
    {
        'linguagem_1': 'Java',
        'linguagem_2': 'Python'
    },
    {
        'linguagem_1': 'JavaScript',
        'linguagem_2': 'Python'
    },
    {
        'linguagem_1': 'JavaScript',
        'linguagem_2': 'TypeScript'
    },
    {
        'linguagem_1': 'Java',
        'linguagem_2': 'C/C++'
    },
    {
        'linguagem_1': 'PHP',
        'linguagem_2': 'JavaScript'
    },
    {
        'linguagem_1': 'PHP',
        'linguagem_2': 'Java'
    },
    {
        'linguagem_1': 'PHP',
        'linguagem_2': 'Python'
    },
    {
        'linguagem_1': 'R',
        'linguagem_2': 'Python'
    },
    {
        'linguagem_1': 'R',
        'linguagem_2': 'JavaScript'
    },
    {
        'linguagem_1': 'R',
        'linguagem_2': 'TypeScript'
    }
]

def verificar_correlacao(linguagem_1, linguagem_2):
    print('Verificando a correlação Pearson entre os % de uso das linguagens: {} e {}.'.format(linguagem_1, linguagem_2))
    corr = df[linguagem_1].corr(df[linguagem_2])
    result = ''
    print(corr)
    if (corr == 1):
        result = 'correlação linear positiva perfeita'
    if (corr > 0):
        result = 'correlação linear positiva'
    if (corr == 0):
        result = 'correlação linear inexistente'
    if (corr == -1):
        result = 'correlação linear negativa perfeita'
    if (corr < 0):
        result = 'correlação linear negativa'
    print('{} e {} possuem {}.\n'.format(linguagem_1, linguagem_2, result))
    
    
def plotar_correlacao(linguagem_1, linguagem_2):
    df.plot.scatter(x = linguagem_1, y = linguagem_2, c = 'Darkblue')

    configurar_plot_com_dimensoes('Correlação entre {} e {}.'.format(linguagem_1, linguagem_2), '', '', 20, 10)
    
for target in targets:
    verificar_correlacao(target['linguagem_1'], target['linguagem_2'])
    
for target in targets:
    plotar_correlacao(target['linguagem_1'], target['linguagem_2'])

df_java = dados_tratados[dados_tratados['Language'] == 'Java']
sns.regplot(x="UnixTime", y="Value", data= df_java)
plt.gcf().set_size_inches(16, 6)
plt.ylabel('% de uso da linguagem Java')
plt.xlabel('Anos em Unix Time de 2004 a 2021')
plt.show()

t = df[['Java', 'Python']]
sns.regplot(x="Java", y="Python", data=t)
plt.gcf().set_size_inches(16, 6)
plt.ylabel('Regrssão do % de uso da linguagem Java e Python')
plt.xlabel('Anos em Unix Time de 2004 a 2021')
plt.show()

# ### Criando os modelos de Machine Learning para o algoritmo de Regressão Linear para as correlações observadas

def prever_regressao_linguagem(linguagem_1, linguagem_2):
    df_linguagem = df[[linguagem_1, linguagem_2]]

    X = df_linguagem[linguagem_1].values.reshape(-1, 1)
    y = df_linguagem[linguagem_2].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    plt.scatter(X_test, y_test,  color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=3)

    plt.gcf().set_size_inches(16, 6)
    
    print('Teste R-quadrado: {}'.format(reg.score(X_test ,y_test)))
    
    plt.title('Regressão linear entre % de uso das linguagens {} e {}.'.format(linguagem_1, linguagem_2))

    plt.show()

for target in targets:
    prever_regressao_linguagem(target['linguagem_1'], target['linguagem_2'])

def prever_regressao_linguagem(linguagem):
    df_linguagem = dados_tratados[dados_tratados['Language'] == linguagem]

    X = df_linguagem.UnixTime.values.reshape(-1, 1)
    y = df_linguagem.Value.values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    plt.scatter(X_test, y_test,  color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=3)

    plt.gcf().set_size_inches(16, 6)
    
    print('Teste R-quadrado: {}'.format(reg.score(X_test ,y_test)))

    plt.title('% de uso da linguagem {}'.format(linguagem))

    plt.show()

prever_regressao_linguagem('Java')
prever_regressao_linguagem('JavaScript')
prever_regressao_linguagem('Python')
prever_regressao_linguagem('TypeScript')
prever_regressao_linguagem('C/C++')
prever_regressao_linguagem('C#')
prever_regressao_linguagem('PHP')
prever_regressao_linguagem('R')

dados = df
dados.head()

dados['Date'] = pd.to_datetime(dados['Date'])
dados.set_index('Date', inplace = True) 

dados.head()

dados.describe()

fig, axes = plt.subplots(nrows=7, ncols=2)

dados['Java'].plot(ax=axes[0,0], title = "Análise da linguagem Java nos últimos 17 anos")
dados['JavaScript'].plot(ax=axes[1,0], title = "Análise da linguagem JavaScript nos últimos 17 anos")
dados['Python'].plot(ax=axes[0,1], title = "Análise da linguagem Python nos últimos 17 anos")
dados['C#'].plot(ax=axes[1,1], title = "Análise da linguagem C# nos últimos 17 anos")
dados['PHP'].plot(ax=axes[2,0], title = "Análise da linguagem PHP nos últimos 17 anos")
dados['Delphi'].plot(ax=axes[2,1], title = "Análise da linguagem Delphi nos últimos 17 anos")
dados['Dart'].plot(ax=axes[3,0], title = "Análise da linguagem Dart nos últimos 17 anos")
dados['Cobol'].plot(ax=axes[3,1], title = "Análise da linguagem Cobol nos últimos 17 anos")
dados['Go'].plot(ax=axes[4,0], title = "Análise da linguagem Go nos últimos 17 anos")
dados['C/C++'].plot(ax=axes[4,1], title = "Análise da linguagem C/C++ nos últimos 17 anos")
dados['Groovy'].plot(ax=axes[5,0], title = "Análise da linguagem Groovy nos últimos 17 anos")
dados['TypeScript'].plot(ax=axes[5,1], title = "Análise da linguagem TypeScript nos últimos 17 anos")
dados['Kotlin'].plot(ax=axes[6,0], title = "Análise da linguagem Kotlin nos últimos 17 anos")
dados['R'].plot(ax=axes[6,1], title = "Análise da linguagem R nos últimos 17 anos")

plt.gcf().set_size_inches(16, 36)

plt.savefig('imgs/Comparação linguagens 17 anos')
plt.show()


# # Analisando dados de 2017

dados_2017 = pd.read_csv('dados/user-languages.csv')    [['user_id', 'java', 'javascript', 'typescript', 'php', 'python', 'c#', 'go']]

dados_2017.head()

dados_2017.dtypes

dados_2017.info()

dados_2017.describe()

dados_2017.hist()
configurar_plot('', '', 'Histograma das linguagens a serem analisadas')

dft = pd.DataFrame(
    {
        'Language': list(map(lambda i: i, dados_2017.columns[1:])),
        'Value':    list(map(lambda i: dados_2017[i].sum(), dados_2017.columns[1:]))
    }
)

dft.head()

dft.describe()

dft.hist()
configurar_plot('Histograma dos dados agrupados', '', '')

dft = dft.sort_values(by=['Value'], ascending=False)

fig, ax = plt.subplots()

ax.set_xticklabels(dft['Language'])

ax.bar(x = dft['Language'], height = dft['Value'])

plt.gcf().set_size_inches(35, 10)

plt.savefig('imgs/Analise Linguagens Dataset 02.png')

configurar_plot_com_dimensoes(
    'Linguagens de programação.',
    '% de uso.',
    'Análise de linguagens mais usadas entre 2004 e 2021.',
    20, 
    8
)

targets = [
    {
        'linguagem_1': 'java',
        'linguagem_2': 'javascript'
    },
    
    {
        'linguagem_1': 'java',
        'linguagem_2': 'python'
    },
    {
        'linguagem_1': 'javascript',
        'linguagem_2': 'python'
    },
    {
        'linguagem_1': 'javascript',
        'linguagem_2': 'typescript'
    },
    {
        'linguagem_1': 'java',
        'linguagem_2': 'php'
    },
    {
        'linguagem_1': 'java',
        'linguagem_2': 'c#'
    },
    {
        'linguagem_1': 'php',
        'linguagem_2': 'python'
    },
    {
        'linguagem_1': 'php',
        'linguagem_2': 'javascript'
    }
]

def verificar_correlacao(linguagem_1, linguagem_2):
    print('Verificando a correlação Pearson entre os % de uso das linguagens: {} e {}.'.format(linguagem_1, linguagem_2))
    corr = dados_2017[linguagem_1].corr(dados_2017[linguagem_2])
    result = ''
    print(corr)
    if (corr == 1):
        result = 'correlação linear positiva perfeita'
    if (corr > 0):
        result = 'correlação linear positiva'
    if (corr == 0):
        result = 'correlação linear inexistente'
    if (corr == -1):
        result = 'correlação linear negativa perfeita'
    if (corr < 0):
        result = 'correlação linear negativa'
    print('{} e {} possuem {}.\n'.format(linguagem_1, linguagem_2, result))
    
    
def plotar_correlacao(linguagem_1, linguagem_2):
    dados_2017.plot.scatter(x = linguagem_1, y = linguagem_2, c = 'Darkblue')
    configurar_plot_com_dimensoes('Correlação entre {} e {}.'.format(linguagem_1, linguagem_2), '', '', 20, 10)

for target in targets:
    verificar_correlacao(target['linguagem_1'], target['linguagem_2'])
    
for target in targets:
    plotar_correlacao(target['linguagem_1'], target['linguagem_2'])

def prever_regressao_linguagem(linguagem_1, linguagem_2):
    df_linguagem = dados_2017[[linguagem_1, linguagem_2]]

    X = df_linguagem[linguagem_1].values.reshape(-1, 1)
    y = df_linguagem[linguagem_2].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    plt.scatter(X_test, y_test,  color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=3)

    plt.gcf().set_size_inches(16, 6)
    
    print('Teste R-quadrado: {}'.format(reg.score(X_test ,y_test)))

    plt.title('Regressão linear entre % de uso das linguagens {} e {}.'.format(linguagem_1, linguagem_2))
    plt.ylabel('% de uso')

    plt.show()

for target in targets:
    prever_regressao_linguagem(target['linguagem_1'], target['linguagem_2'])

# # Analisando o dataset do Stackoverflow

# ### Análise da Query no Stackoverflow:
# 
# https://data.stackexchange.com/stackoverflow/query
# 
# ```sql
# SELECT 
#   YEAR(Posts.CreationDate) as 'Year',
#   MONTH(Posts.CreationDate) as 'Month',
#   Tags.tagName,
#   COUNT(*) AS Question
# FROM Tags
#   LEFT JOIN PostTags ON PostTags.TagId = Tags.Id
#   LEFT JOIN Posts ON Posts.Id = PostTags.PostId
# WHERE
#   Tags.tagName IN (
#     'java',
#     'javascript',
#     'typescript',
#     'python',
#     'go',
#     'c#', 
#     'c', 
#     'c++', 
#     'php',
#     'r'
#   )
#   AND Posts.CreationDate <= '2021-12-31'
# GROUP BY
#   YEAR(Posts.CreationDate), MONTH(Posts.CreationDate), Tags.TagName
# ORDER BY 
#   YEAR(Posts.CreationDate), MONTH(Posts.CreationDate) DESC
# ```

df_so = pd.read_csv('dados/stack_overflow/QueryResults.csv')

df_so.head()

df_so.info()

df_so = df_so.dropna()

df_so.tail()

df_so.describe()

df_so.Year

def tratar_nome(nome):
    if (nome == 'java'):
        return 'Java'
    
    if (nome == 'javascript'):
        return 'JavaScript'
    
    if (nome == 'typescript'):
        return 'TypeScript'
    
    if (nome == 'python'):
        return 'Python'
    
    if (nome == 'php'):
        return 'PHP'
    
    if (nome == 'c'):
        return 'C/C++'
    
    if (nome == 'c++'):
        return 'C/C++'
    
    if (nome == 'c#'):
        return 'C#'
    
    if (nome == 'go'):
        return 'Go'
    
    if (nome == 'r'):
        return 'R'
    
    return ''

dados_tratados_so = pd.DataFrame(
    {
        'Year': df_so['Year'],
        'Month': df_so['Month'],
        'Language': list(map(lambda x: tratar_nome(x), df_so['tagName'])),
        'Value': df_so['Question']
    }
)

dados_tratados_so.head()

dados_tratados_so_gb = dados_tratados_so    .groupby(by=['Language'], as_index=False)    .sum()    .sort_values(by=['Value'], ascending=False)

fig, ax = plt.subplots()

ax.set_xticklabels(dados_tratados_so_gb['Language'])

ax.bar(x = dados_tratados_so_gb['Language'], height = dados_tratados_so_gb['Value'])

plt.gcf().set_size_inches(35, 10)

plt.savefig('imgs/Analise Linguagens Dataset 02.png')

configurar_plot_com_dimensoes(
    'Análise de linguagens mais usadas entre 2008 e 2021.',
    'Linguagens de programação.',
    '% de uso.',
    20, 
    8
)

def find_by_date_and_language(year, month, language):
    value = dados_tratados_so[
        (dados_tratados_so['Language'] == language) 
        & (dados_tratados_so['Year'] == year)
        & (dados_tratados_so['Month'] == month)
    ]['Value']
    if (value.empty):
        return 0
    return value.values[0]

transposed_data = []

for year in dados_tratados_so['Year'].unique():
    for month in dados_tratados_so['Month'].unique():
        data = {
            'Year': year,
            'Month': month,
            'JavaScript': find_by_date_and_language(year, month, 'JavaScript'),
            'TypeScript': find_by_date_and_language(year, month, 'TypeScript'),
            'Python': find_by_date_and_language(year, month, 'Python'),
            'Java': find_by_date_and_language(year, month, 'Java'),
            'C#': find_by_date_and_language(year, month, 'C#'),
            'PHP': find_by_date_and_language(year, month, 'PHP'),
            'C/C++': find_by_date_and_language(year, month, 'C/C++'),
            'R': find_by_date_and_language(year, month, 'R'),
            'Go': find_by_date_and_language(year, month, 'Go')
        }
        transposed_data.append(data)

correlation_df = pd.DataFrame(transposed_data, columns =[
    'Year',
    'Month', 
    'JavaScript',
    'TypeScript',
    'Python',
    'Java',
    'C#',
    'PHP',
    'C/C++',
    'R',
    'Go'
])
correlation_df.tail(10)

targets = [
    {
        'linguagem_1': 'Java',
        'linguagem_2': 'JavaScript'
    },
    {
        'linguagem_1': 'Java',
        'linguagem_2': 'Python'
    },
    {
        'linguagem_1': 'JavaScript',
        'linguagem_2': 'Python'
    },
    {
        'linguagem_1': 'JavaScript',
        'linguagem_2': 'TypeScript'
    },
    {
        'linguagem_1': 'Java',
        'linguagem_2': 'C/C++'
    },
    {
        'linguagem_1': 'PHP',
        'linguagem_2': 'JavaScript'
    },
    {
        'linguagem_1': 'PHP',
        'linguagem_2': 'Java'
    },
    {
        'linguagem_1': 'PHP',
        'linguagem_2': 'Python'
    },
    {
        'linguagem_1': 'R',
        'linguagem_2': 'Python'
    },
    {
        'linguagem_1': 'R',
        'linguagem_2': 'JavaScript'
    },
    {
        'linguagem_1': 'R',
        'linguagem_2': 'TypeScript'
    }
]

def verificar_correlacao(linguagem_1, linguagem_2):
    print('Verificando a correlação Pearson entre os % de uso das linguagens: {} e {}.'.format(linguagem_1, linguagem_2))
    corr = correlation_df[linguagem_1].corr(correlation_df[linguagem_2])
    result = ''
    print(corr)
    if (corr == 1):
        result = 'correlação linear positiva perfeita'
    if (corr > 0):
        result = 'correlação linear positiva'
    if (corr == 0):
        result = 'correlação linear inexistente'
    if (corr == -1):
        result = 'correlação linear negativa perfeita'
    if (corr < 0):
        result = 'correlação linear negativa'
    print('{} e {} possuem {}.\n'.format(linguagem_1, linguagem_2, result))
    
    
def plotar_correlacao(linguagem_1, linguagem_2):
    correlation_df.plot.scatter(x = linguagem_1, y = linguagem_2, c = 'Darkblue')

    configurar_plot_com_dimensoes('Correlação entre {} e {}.'.format(linguagem_1, linguagem_2), '', '', 20, 10)

for target in targets:
    verificar_correlacao(target['linguagem_1'], target['linguagem_2'])
    
for target in targets:
    plotar_correlacao(target['linguagem_1'], target['linguagem_2'])

def prever_regressao_linguagem(linguagem_1, linguagem_2):
    df_linguagem = correlation_df[[linguagem_1, linguagem_2]]

    X = df_linguagem[linguagem_1].values.reshape(-1, 1)
    y = df_linguagem[linguagem_2].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    plt.scatter(X_test, y_test,  color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=3)

    plt.gcf().set_size_inches(16, 6)
    
    print('Teste R-quadrado: {}'.format(reg.score(X_test ,y_test)))
    
    plt.title('Regressão linear entre % de uso das linguagens {} e {}.'.format(linguagem_1, linguagem_2))
    plt.ylabel('% de uso')

    plt.show()

for target in targets:
    prever_regressao_linguagem(target['linguagem_1'], target['linguagem_2'])

# ### Realizando o Merge com os dados do Github

dados_tratados.head()

def extrair_mes_data(data):
    if ('January' in data):
        return 1
    if ('February' in data):
        return 2
    if ('March' in data):
        return 3
    if ('April' in data):
        return 4
    if ('May' in data):
        return 5
    if ('June' in data):
        return 6
    if ('July' in data):
        return 7
    if ('August' in data):
        return 8
    if ('September' in data):
        return 9
    if ('October' in data):
        return 10
    if ('November' in data):
        return 11
    if ('December' in data):
        return 12

dados_tratados_novo = pd.DataFrame(
    {
        'Year': dados_tratados['Year'],
        'Month': list(map(lambda x: extrair_mes_data(x), dados_tratados['Date'])),
        'Language': dados_tratados['Language'],
        'Value': dados_tratados['Value']
    }
)

dados_tratados_novo = dados_tratados_novo[
    (dados_tratados_novo['Language'] == 'Java') |
    (dados_tratados_novo['Language'] == 'JavaScript') |
    (dados_tratados_novo['Language'] == 'TypeScript') |
    (dados_tratados_novo['Language'] == 'Python') |
    (dados_tratados_novo['Language'] == 'R') |
    (dados_tratados_novo['Language'] == 'C/C++') |
    (dados_tratados_novo['Language'] == 'C#') |
    (dados_tratados_novo['Language'] == 'PHP') |
    (dados_tratados_novo['Language'] == 'Go')
]

merge = pd.merge(dados_tratados_novo, dados_tratados_so, how='left', on=['Year', 'Month', 'Language'])
merge = merge.dropna()
merge.head()

def prever_regressao_linguagem(linguagem):
    df_linguagem = merge[merge['Language'] == linguagem]

    X = df_linguagem['Value_x'].values.reshape(-1, 1)
    y = df_linguagem['Value_y'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    
    plt.scatter(X_test, y_test,  color='blue')
    plt.plot(X_test, y_pred, color='red', linewidth=3)

    plt.gcf().set_size_inches(16, 6)
    
    print('Teste R-quadrado: {}'.format(reg.score(X_test ,y_test)))

    plt.ylabel('Valor StackOverFlow')
    plt.xlabel('Valor GitHub')
    plt.title('Regressão linear entre os valores do StackOverflow e do % uso do Github para linguagem {}'.format(linguagem))

    plt.show()
    
prever_regressao_linguagem('Java')
prever_regressao_linguagem('JavaScript')
prever_regressao_linguagem('TypeScript')
prever_regressao_linguagem('Python')
prever_regressao_linguagem('Go')
prever_regressao_linguagem('PHP')
prever_regressao_linguagem('C/C++')
prever_regressao_linguagem('C#')
prever_regressao_linguagem('R')

# ### Instanciando o algoritmo K-Nearest Neighbors para os valores do Github e StackOverflow

X = merge.Value_x.values.reshape(-1, 1)
y = merge.Language.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
y_pred

knn.predict_proba(X_test)

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

X = merge.Value_y.values.reshape(-1, 1)
y = merge.Language.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ### Instanciando o algoritmo Naive Bayes para os valores do Github e StackOverflow

X = merge.Value_y.values.reshape(-1, 1)
y = merge.Language.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

naive_bayes = GaussianNB()

pred = naive_bayes.fit(X_train, y_train)

y_pred = pred.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

X = merge.Value_y.values.reshape(-1, 1)
y = merge.Language.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

naive_bayes = GaussianNB()

pred = naive_bayes.fit(X_train, y_train)

y_pred = pred.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ### Instanciando o algoritmo Support Vector Machines (SVM) para os valores do Github e StackOverflow

X = merge.Value_x.values.reshape(-1, 1)
y = merge.Language.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = svm.SVC()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

X = merge.Value_y.values.reshape(-1, 1)
y = merge.Language.values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = svm.SVC()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def realizar_previsoes(algoritmo, base, use_scalling):
    target = []
    
    if (base == 'github'):
        target = merge.Value_x
    else:
        target = merge.Value_y
    
    y = merge.Language.values.reshape(-1, 1)
    X = target.values.reshape(-1, 1)
    
    if (use_scalling):
        X = preprocessing.StandardScaler().fit(X).transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    y_pred = []
    
    if (algoritmo == 'knn'):
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
    
    if (algoritmo == 'naive_bayes'):
        naive_bayes = GaussianNB()
        naive_bayes.fit(X_train, y_train)
        y_pred = pred.predict(X_test)
        
    if (algoritmo == 'svm'):
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
    
    score = accuracy_score(y_test, y_pred) 
    classificacao = classification_report(y_test, y_pred) 
    
    print('Algoritmo: {}\nDados do: {}\nScore: {}\nRelatório de classificação:'.format(algoritmo, base, score))
    print(classificacao)
    return score

# ### Realizando previsões com normalização

github_knn = realizar_previsoes('knn', 'github', True)
github_naive_bayes = realizar_previsoes('naive_bayes', 'github', True)
github_svm = realizar_previsoes('svm', 'github', True)

stackoverflow_knn = realizar_previsoes('knn', 'stackoverflow', True)
stackoverflow_naive_bayes = realizar_previsoes('naive_bayes', 'stackoverflow', True)
stackoverflow_svm = realizar_previsoes('svm', 'stackoverflow', True)

resultados = pd.DataFrame({
    'algoritmo': ['KNN', 'Naive Bayes', 'SVM'],
    'github': [github_knn, github_naive_bayes, github_svm],
    'stackoverflow': [stackoverflow_knn, stackoverflow_naive_bayes, stackoverflow_svm]
})

print(resultados.head())

resultados.plot(kind = 'bar', x='algoritmo')
configurar_plot('Análise de score x algoritmo', 'algoritmo', 'score')

# ### Realizando previsões sem normalização

github_knn = realizar_previsoes('knn', 'github', False)
github_naive_bayes = realizar_previsoes('naive_bayes', 'github', False)
github_svm = realizar_previsoes('svm', 'github', False)

stackoverflow_knn = realizar_previsoes('knn', 'stackoverflow', False)
stackoverflow_naive_bayes = realizar_previsoes('naive_bayes', 'stackoverflow', False)
stackoverflow_svm = realizar_previsoes('svm', 'stackoverflow', False)

resultados = pd.DataFrame({
    'algoritmo': ['KNN', 'Naive Bayes', 'SVM'],
    'github': [github_knn, github_naive_bayes, github_svm],
    'stackoverflow': [stackoverflow_knn, stackoverflow_naive_bayes, stackoverflow_svm]
})

print(resultados.head())

resultados.plot(kind = 'bar', x='algoritmo')
configurar_plot('Análise de score x algoritmo', 'algoritmo', 'score')

# ### Realizando previsões com normalização em KNN e SVM, e sem normalização em GNB

github_knn = realizar_previsoes('knn', 'github', True)
github_naive_bayes = realizar_previsoes('naive_bayes', 'github', False)
github_svm = realizar_previsoes('svm', 'github', True)

stackoverflow_knn = realizar_previsoes('knn', 'stackoverflow', True)
stackoverflow_naive_bayes = realizar_previsoes('naive_bayes', 'stackoverflow', False)
stackoverflow_svm = realizar_previsoes('svm', 'stackoverflow', True)

resultados = pd.DataFrame({
    'algoritmo': ['KNN', 'Naive Bayes', 'SVM'],
    'github': [github_knn, github_naive_bayes, github_svm],
    'stackoverflow': [stackoverflow_knn, stackoverflow_naive_bayes, stackoverflow_svm]
})

print(resultados.head())

resultados.plot(kind = 'bar', x='algoritmo')
configurar_plot('Análise de score x algoritmo', 'algoritmo', 'score')