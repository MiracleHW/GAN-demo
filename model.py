import tensorflow as tf
import tensorflow.contrib.slim as slim

BATCH_SIZE = 64


def generator(z, train=True,reuse=None):
    with tf.variable_scope("g_",reuse=reuse):
        #h1 = tf.nn.relu(batch_norm_layer(fully_connected(z, 1024, 'g_fully_connected1'),is_train=train, name='g_bn1'))

        h1=slim.fully_connected(z,1024,activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,scope="fc1")

        #h2 = tf.nn.relu(batch_norm_layer(fully_connected(h1, 128 * 49, 'g_fully_connected2'),is_train=train, name='g_bn2'))
        h2=slim.fully_connected(z,128*49,activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,scope="fc2")

        h2 = tf.reshape(h2, [64, 7, 7, 128], name='h2_reshape')

        #h3 = tf.nn.relu(batch_norm_layer(deconv2d(h2, [64, 14, 14, 128],name='g_deconv2d3'),is_train=train, name='g_bn3'))
        h3=slim.conv2d_transpose(h2,128,[5,5],stride=2,activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,scope="deconv1")

        #h4 = tf.nn.sigmoid(deconv2d(h3, [64, 28, 28, 1],name='g_deconv2d4'), name='generate_image')
        h4=slim.conv2d_transpose(h3,1,[5,5],stride=2,activation_fn=tf.nn.sigmoid,scope="deconv2")

    return h4


def discriminator(x,  reuse=None):
    with tf.variable_scope("d_",reuse=reuse):
        #h1 = lrelu(conv2d(x, 11, name='d_conv2d1'), name='lrelu1')
        h1=slim.conv2d(x,11,[5,5],stride=2,activation_fn=lrelu,scope="conv1")

        #h2 = lrelu(batch_norm_layer(conv2d(h1, 74, name='d_conv2d2'),name='d_bn2'), name='lrelu2')
        h2=slim.conv2d(h1,74,[5,5],stride=2,activation_fn=lrelu,normalizer_fn=slim.batch_norm,scope="conv2")
        h2=tf.reshape(h2,[BATCH_SIZE,-1])

        #h3 = lrelu(batch_norm_layer(fully_connected(h2, 1024, name='d_fully_connected3'),name='d_bn3'), name='lrelu3')
        h3=slim.fully_connected(h2,1024,activation_fn=lrelu,normalizer_fn=slim.batch_norm,scope="fc1")

        #h4 = fully_connected(h3, 1, name='d_result_withouts_sigmoid')
        h4=slim.fully_connected(h3,1,activation_fn=tf.nn.sigmoid,scope="fc2")

        return tf.nn.sigmoid(h4, name='discriminator_result_with_sigmoid')


def sampler(z,  train=True):
    return generator(z, train=train,reuse=True)

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x, name=name)
