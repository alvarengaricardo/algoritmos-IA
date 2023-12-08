# Importando bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Criando dados de exemplo com 20 posições
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y = np.array([2, 4, 5, 2, 5, 7, 8, 9, 10, 11, 13, 18, 14, 16, 16, 17, 18, 19, 19, 20])

# Reshape para o formato esperado pelo modelo
x = x.reshape(-1, 1)

# Criando o modelo de regressão linear
model = LinearRegression()

# Treinando o modelo
model.fit(x, y)

# Obtendo os coeficientes
m = model.coef_[0]
b = model.intercept_

# Prevendo valores com o modelo treinado
y_pred = model.predict(x)

# Plotando os dados e a linha de regressão
plt.scatter(x, y, color='blue', label='Dados Observados')
plt.plot(x, y_pred, color='red', linewidth=2, label='Regressão Linear')
plt.xlabel('Variável Independente (x)')
plt.ylabel('Variável Dependente (y)')
plt.title('Regressão Linear Simples com 20 Posições')
plt.legend()
plt.show()

# Exibindo os coeficientes
print(f'Inclinação (m): {m}')
print(f'Intercepção (b): {b}')
