import numpy as np

def prediction_signle_loop(x,w,b):
    n=x.shape[0]
    p=0
    for i in range(n):
        p=p+x[i]*w[i]
    p=p+b
    return p;
# single varialbe in the funtion
def single_predicntion(x,w,b):
    p=np.dot(x,w)+b
    return p


#  cost function
def cost_c(x,y,m,b):
    m=x.shape[0]
    cost=0.0
    for i in range(m):
        f_wb_i=np.dot(x[i],w)+b
        cost=cost+(f_wb_i-y[i])**2
    cost=cost/(2*m)
    return cost
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
# print(X_train.shape)
b=785.658
w=np.array([0.325,18.355,-53.5265,-26.425])

x_vec=X_train[0,:]
f_wb=prediction_signle_loop(x_vec,w,b)
print(f_wb)
print(single_predicntion(x_vec,w,b))


#  compute the cost function in the lenear Regrasseion function
print(cost_c(X_train,y_train,w,b))