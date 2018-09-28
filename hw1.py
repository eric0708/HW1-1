import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size]))
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs
  
#create data
x_data=np.linspace(-1,1,300)[:,np.newaxis]
y_data=np.add(np.power(np.sin(np.multiply(8,x_data)),3),np.multiply(4,x_data))
#y_data=np.exp(np.sin(np.multiply(6,x_data)))

#define placeholders for the inputs
xs=tf.placeholder(tf.float32,[None,1])
ys=tf.placeholder(tf.float32,[None,1])

#add hidden layer
l1=add_layer(xs,1,6,activation_function=tf.nn.relu)
l12=add_layer(xs,1,34,activation_function=tf.nn.relu)
l2=add_layer(l1,6,6,activation_function=tf.nn.relu)
l3=add_layer(l2,6,6,activation_function=tf.nn.relu)

prediction=add_layer(l3,6,1,activation_function=None)
prediction2=add_layer(l12,34,1,activation_function=None)
#Error
loss=tf.reduce_mean(tf.reduce_sum(tf.square(prediction-y_data),reduction_indices=[1]))
loss2=tf.reduce_mean(tf.reduce_sum(tf.square(prediction2-y_data),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step2=tf.train.GradientDescentOptimizer(0.1).minimize(loss2)


init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)


fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(x_data,y_data,'b-',lw=2)
plt.ion()
plt.show()

       
for i in range(20000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    sess.run(train_step2,feed_dict={xs:x_data,ys:y_data})
    if i%40==0:
        try:
            ax.lines.remove(lines[0])
            ax.lines.remove(lines1[0])
        except Exception:
            pass
        prediction_value=sess.run(prediction,feed_dict={xs:x_data})
        prediction_value2=sess.run(prediction2,feed_dict={xs:x_data})
        lines=ax.plot(x_data,prediction_value,'r-',lw=2)
        lines1=ax.plot(x_data,prediction_value2,'g-',lw=2)
        plt.pause(0.01)
        print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
        print(sess.run(loss2, feed_dict={xs:x_data, ys:y_data}))

print("variable count",np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
input()
