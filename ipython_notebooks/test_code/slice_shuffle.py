import numpy as np

X = np.arange(100)
Z = np.arange(100,200)

Y = np.linspace(0,1,100)
print(X)
print(len(Y), len(X))

Y = X[15:30]
print(Y)

rnd_state = np.random.get_state()
X = np.random.permutation(X)
np.random.set_state(rnd_state)
Z = np.random.permutation(Z)
print(Y)
#Y state hasn't changed;
print(X-Z)
#notice that X-Z is constant :D

#reset;
X = np.arange(100)
Z = np.arange(100,200)
X = np.reshape(X,(10,-1))

rnd_state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(rnd_state)
np.random.shuffle(Z)
Z = np.reshape(Z,(10,-1))
print(X-Z)
#notice that X-Z is no longer constant
# reasoning (guess): shape plays a roll in the data order
#                  : shuffle just shuffles all elements in place


#reset;
X = np.arange(100)
Z = np.arange(100,200)
print(X)
Y = X[15:30]
print(Y)
print("X[Y] = ", X[Y])
np.random.shuffle(X)
print(X)
print(Y)
print("X[Y] = ", X[Y])
#Y state hasn't changed
#X[Y] state has changed


