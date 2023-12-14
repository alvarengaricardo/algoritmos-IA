# Importando bibliotecas necessárias
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Criando dados de exemplo
np.random.seed(42)
X = np.random.rand(100, 1)
y = (X.squeeze() + 0.2 * np.random.randn(100)) > 0.5

# Criando o modelo de regressão logística
model = LogisticRegression()
model.fit(X, y)

# Gerando valores para previsão
X_test = np.linspace(0, 1, 400)
y_prob = model.predict_proba(X_test.reshape(-1, 1))[:, 1]

# Plotando os dados e a curva de decisão
plt.scatter(X, y, color='blue', label='Dados Observados')
plt.plot(X_test, y_prob, color='red', linewidth=2, label='Curva de Decisão')
plt.xlabel('Variável Independente (X)')
plt.ylabel('Probabilidade de Classe 1')
plt.title('Regressão Logística')
plt.legend()
plt.show()
