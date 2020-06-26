
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


plt.scatter(X[:, 0], X[:, 1], c=y, marker=".",
            cmap=matplotlib.cm.get_cmap(name="bwr"), alpha=0.7)


Xi = np.linspace(-10, 10)
Y = -model.coef_[0][0] / model.coef_[0][1] * \
    Xi - model.intercept_ / model.coef_[0][1]
plt.plot(Xi, Y)


plt.xlim(min(X[:, 0]) - 0.5, max(X[:, 0]) + 0.5)
plt.ylim(min(X[:, 1]) - 0.5, max(X[:, 1]) + 0.5)
plt.axes().set_aspect("equal", "datalim")


plt.title("classification data using LogisticRegression")

plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.show()


#x,y 가 없는데뭐하라는지 모르겠다.