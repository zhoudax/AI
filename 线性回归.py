%matplotlib inline
import matplot.pyplot as plt
import numpy as np
import tensorflow as tf
tf.compat.v1.disabel_eager_execution()
np.random.seed(5)
x_data=np.linspace(-1,1,100)
y_data=2*x_data+1+np.random.randn(*x_data.shape)*4
plt.scatter(x_data,y_data)
x=tf.compat.v1.placeholder()
