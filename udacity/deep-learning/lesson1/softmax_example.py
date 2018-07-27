""" Softmax """

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for x."""
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=0)
    return np.divide(ex, sum_ex)


scores = np.array([3.0, 1.0, 0.2])
print(scores)
print(softmax(scores))
print(softmax(scores*10.0))
print(softmax(scores/10.0))


# Print softmax curves:
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

