import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Dados de treinamento
X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 1, 1, 0])

# Criação do modelo
clf = svm.SVC(kernel='linear')

# Treinamento do modelo
clf.fit(X, y)

# Previsão de novos valores
print(clf.predict([[2., 2.], [-1., -1.]]))

# Plot dos dados
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# Plot da reta de separação
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Criação da malha para plotagem
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Plot da margem e das fronteiras de decisão
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')
plt.show()
