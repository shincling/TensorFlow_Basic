import tensorflow as tf
m1=tf.constant([[3.,3.]])#1*2
m2=tf.constant([[2.],[2.]])#2*1
product=tf.matmul(m1,m2)

sess=tf.Session()
result=sess.run(product)
print result

sess.close()

