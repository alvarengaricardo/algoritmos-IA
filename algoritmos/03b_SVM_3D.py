# Importando bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Carregando o conjunto de dados de câncer de mama
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

# Padronizando os dados e reduzindo a dimensionalidade com PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Criando e treinando um modelo SVM
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Realizando previsões no conjunto de teste
predictions = svm_model.predict(X_test)

# Avaliando a acurácia do modelo
accuracy = accuracy_score(y_test, predictions)
print(f'Acurácia do SVM: {accuracy}')

# ...

# Visualizando a fronteira de decisão em 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

h = .02  # Passo da grade no espaço de atributos
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
z_min, z_max = X_pca[:, 2].min() - 1, X_pca[:, 2].max() + 1

# Corrigindo a geração da malha 3D
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h))

# Colocando o resultado no espaço de atributos
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
Z = Z.reshape(xx.shape)

# Criando o gráfico de contorno em 3D
ax.contourf(xx, yy, zz, Z, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

ax.set_title('Fronteira de Decisão do SVM após Redução de Dimensionalidade PCA (3D)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')

plt.show()

