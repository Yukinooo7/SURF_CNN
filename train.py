from get_dataset import load_mnist
import tensorflow as tf
import input_data

if __name__ == "__main__":
    mnist = input_data.read_data_sets("E:\\Study\\2019-Summer\\SURF\\CNN\\MNIST_data\\", one_hot=True)

    # Set placeholder for train images
    x = tf.placeholder("float", [None, 784])

    # Set up weight and bias
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # define softmax
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # this is real labels
    y_ = tf.placeholder("float", [None, 10])

    # to calculate the accuracy
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # define what to do on train step
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # Start to train
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    print("Start to train")
    # Training process
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)  # 使用minibatch的训练数据，一个batch的大小为100
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # 用训练数据替代占位符来执行训练
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # tf.argmax()返回的是某一维度上其数据最大所在的索引值，在这里即代表预测值和真值
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 用平均值来统计测试准确率

        print(i, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    sess.close()