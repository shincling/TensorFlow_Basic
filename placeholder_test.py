import tensorflow as tf
# tf.device("/gpu:1")
input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)
option=tf.multiply(input1,input2)

with tf.Session() as sess:
    result=sess.run(option,feed_dict={input1:[7],input2:3})
    print type(result)
    print result

