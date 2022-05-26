import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from IPython.display import clear_output

class neural_layer():

	def __init__(self, n_conn, n_neur, act_f):
		self.act_f = act_f
		self.b = np.random.rand(1,n_neur) * 2 -1
		self.w = np.random.rand(n_conn,n_neur) * 2 -1


def create_nn(topology, act_f):

	nn = []
	for l in range(len(topology[:-1])):
		nn.append(neural_layer(topology[l], topology[l+1], act_f))
	return nn

def train(neural_net, X, Y, cost_f, lr=0.5, train=True):

	out = [(None, X)]

	#Forward Pass

	for l in range(len(neural_net)):

		z = out[-1][1] @ neural_net[l].w + neural_net[l].b
		a = neural_net[l].act_f[0](z)

		out.append((z,a))

	if train:

		#Back Propagation
		deltas = []

		for l in reversed(range(len(neural_net))):

			z = out[l+1][0]
			a = out[l+1][1]

			if l == len(neural_net)-1:
				#calculate delta last layer
				deltas.insert(0, cost_f[1](a, Y) * neural_net[l].act_f[1](a))
			else:
				#calculate delta in respect of the previous layer
				deltas.insert(0, deltas[0] @ _w.T * neural_net[l].act_f[1](a))

			_w = neural_net[l].w

			#Gradient Descent
			neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
			neural_net[l].w = neural_net[l].w - out[l][1].T @ deltas[0] * lr

	return out[-1][1]


#Main Program


#Create DataSet
n = 500

X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)

Y = Y[:, np.newaxis]

plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], c="peru")
plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], c="skyblue")
plt.show()

#Activation functions
sigm = (lambda x: 1 / (1 + np.e **(-x)),
		lambda x: x*(1-x))

relu = lambda x: np.maximum(0,x)

_x = np.linspace(-5, 5, 100)
plt.plot(_x,sigm[0](_x))
plt.show()



topology = [2, 4, 8, 1]
neural_net = create_nn(topology, sigm)

cost_f = (lambda Yp, Yr: np.mean((Yp - Yr)**2),
		  lambda Yp, Yr: (Yp -Yr))

loss = []
res=50

_x0 = np.linspace(-1.5,1.5,res)
_x1 = np.linspace(-1.5,1.5,res)
_y = np.zeros((res,res))

for i in range(2000):
	
	loss.append(cost_f[0](train(neural_net, X, Y, cost_f, 0.05), Y))
	
	if i % 50 == 0:
		
		for i0,x0 in enumerate(_x0):
			for i1,x1 in enumerate(_x1):
				_y[i0,i1] = train(neural_net,np.array([x0,x1]),Y, cost_f,train=False)[0][0]
		plt.pcolormesh(_x0,_x1,_y, cmap="BrBG")
		plt.axis("equal")

		plt.scatter(X[Y[:,0]==0,0], X[Y[:,0]==0,1], c="peru")
		plt.scatter(X[Y[:,0]==1,0], X[Y[:,0]==1,1], c="skyblue")

		clear_output(wait=True)

		if loss[-1] < 0.005:
			plt.show()
			plt.plot(range(len(loss)),loss)
			plt.show()
			break

		plt.show(block=False)
		plt.pause(1)
		plt.close()
		plt.plot(range(len(loss)),loss)
		plt.show(block=False)
		plt.pause(1)
		plt.close()