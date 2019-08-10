from os import walk
from os.path import join
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt


def read_images(path):
    """从源文件/路径读取图像

    参数：
        path: 图像所在的路径即文件夹名称
    返回:
        返回一个带有所有图像、标签和总数信息的对象
        images: 所有的图像数据
        labels: 所有标签
        num: 数目
    """

    # 获取文件夹内所有图像文件的文件名和总数
    filenames = next(walk(path))[2]
    num_file = len(filenames)

    # 初始化图像和标签
    images = np.zeros((num_file, 300, 300, 3), dtype=np.uint8)
    labels = np.zeros((num_file,), dtype=np.uint8)

    # 遍历读取文件
    for index, filename in enumerate(filenames):
        # 读取单张图像，并且修改为自定义尺寸
        img = Image.open(join(path, filename))
        img = img.crop((100, 30, 400, 330))  # 300 * 300 cropped image
        images[index] = img

        # TO DO
        # 这里通过文件名获取标签信息，猫狗大战问题中只有两类，故只有0和1
        # 可以根据自己的需要进行修改
        # 注意：这里不是one-hot编码
        if filename[0:3] == 'cat':
            labels[index] = int(0)
        else:
            labels[index] = int(1)

        if index % 1000 == 0:
            print("Reading the %sth image" % index)

    # 创建一个类，该类携带图像、标签和总数信息
    class ImgData(object):
        pass

    result = ImgData()
    result.images = images
    result.labels = labels
    result.num = num_file

    return result


def convert(data, destination):
    """将图片存储为.tfrecords文件

    参数:
        data: 上述函数返回的ImageData对象
        destination: 目标文件名
    """

    images = data.images
    labels = data.labels
    num_examples = data.num

    # 存储的文件名
    filename = destination

    # 使用TFRecordWriter来写入数据
    writer = tf.python_io.TFRecordWriter(filename)
    # 遍历图片
    for index in range(num_examples):
        # 转为二进制
        image = images[index].tostring()
        label = labels[index]
        # tf.train下有Feature和Features，需要注意其区别
        # 层级关系为Example->Features->Feature
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        # 写入
        writer.write(example.SerializeToString())
    writer.close(

    )


def read_and_decode(filename_queue):
    """读取.tfrecords文件

            参数:
                filename_queue: 文件名, 一个列表

            返回:
                img, label: **单张图片和对应标签**
            """
    # 创建一个图节点，该节点负责数据输入
    filename_queue = tf.train.string_input_producer([filename_queue])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 解析单个example
    features = tf.parse_single_example(serialized_example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [300, 300, 3])
    image = tf.cast(image, tf.float32)
    label = tf.cast(features['label'], tf.int64)

    return image, label


def distorted_input(filename, batch_size):
    """建立一个乱序的输入

    参数:
      filename: tfrecords文件的文件名. 注：该文件名仅为文件的名称，不包含路径和后缀
      batch_size: 每次读取的batch size

    返回:
      images: 一个4D的Tensor. size: [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3]
      labels: 1D的标签. size: [batch_size]
    """
    image, label = read_and_decode(filename)
    # 乱序读入一个batch
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                            num_threads=16, capacity=3000, min_after_dequeue=1000)

    return images, labels


if __name__ == '__main__':
    # test_result = read_images('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\kaggle\\train')
    # convert(test_result, 'E:\\Study\\2019-Summer\\SURF\\CNN\\data\\kaggle\\tfrecords\\file.tfrecords')
    images, labels = distorted_input('E:\\Study\\2019-Summer\\SURF\\CNN\\data\\kaggle\\tfrecords\\file.tfrecords', batch_size=4)

    print('Done')
    # from matplotlib import pyplot as plt
    fig = plt.figure()
    a = fig.add_subplot(221)
    b = fig.add_subplot(222)
    c = fig.add_subplot(223)
    d = fig.add_subplot(224)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        # 开启文件读取队列，开启后才能开始读取数据
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        img, label = sess.run([images, labels])

        a.imshow(img[0])
        a.axis('off')

        b.imshow(img[1])
        b.axis('off')

        c.imshow(img[2])
        c.axis('off')

        d.imshow(img[3])
        d.axis('off')

        plt.show()

        coord.request_stop()
        coord.join(threads)

