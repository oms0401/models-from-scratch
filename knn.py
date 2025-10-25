"""
The Core Idea 🧠
Imagine you have a scatter plot with blue dots and red dots. Now, you get a new, gray dot and you have to guess its color. What do you do?

Find its neighbors: You'd look at the dots closest to the new gray one.

Hold a vote: You'd count how many of its k closest neighbors are red and how many are blue. If you chose k=5 and 4 of the 5 closest dots are red, you'd bet the gray dot is also red.

That's it. That's the entire logic of KNN. It's a lazy learner because it doesn't really "learn" a model from the training data. It just memorizes the entire dataset and uses it directly during prediction.


"""

from collections import Counter
import numpy as np 

def eucliedean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    
    def predict(self,X):
        y_pred=[self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances=[eucliedean_distance(x,x_train) for x_train in self.X_train]
        k_idx=np.argsort(distances)[:self.k]
        k_labels=[self.y_train[i] for i in k_idx]
        label=Counter(k_labels).most_common(1)

        return label[0][0]

