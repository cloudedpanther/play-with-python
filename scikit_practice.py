import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
plt.style.use(['seaborn-whitegrid'])

x = 10 * np.random.rand(50)
y = 2 * x + np.random.rand(50)

model = LinearRegression(fit_intercept=True)

X = x[:, np.newaxis]

model.fit(X, y)

xfit = np.linspace(-1, 11)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)

plt.scatter(x, y)
plt.plot(xfit, yfit, '--r')
plt.show()