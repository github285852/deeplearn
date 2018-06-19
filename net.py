import numpy as np
import time
D_IN = 12
D_OUT = 1
H = 1000
N=4

D_OUT =   1

W1 = np.random.rand(D_IN,H)
W2 = np.random.rand(H,D_OUT)

learning_rate =0.00000000001 
def neurl_net(X,Y=None):
    global W1
    global W2
    global learning_rate 
    h = X.dot(W1)
    h_relu = np.maximum(h,0)
    y_pre = h_relu.dot(W2)
    loss = np.square(y_pre-y).sum()
    if Y == None:
        return y_pre
    print(loss)
    
    grad_y_pre = 2.0*(y_pre - y)
    grad_W2 = h_relu.T.dot(grad_y_pre)
    grad_h_relu = grad_y_pre.dot(W2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h<0] = 0
    grad_W1 = X.T.dot(grad_h)

    W1 -= learning_rate*grad_W1
    W2 -= learning_rate*grad_W2

t = time.time()
for i in range(5000000):
    x = np.random.rand(N,D_IN)
    y = x.max(axis=1)
    y.shape=(N,1)
    neurl_net(x,y)
print t
print time.time()
