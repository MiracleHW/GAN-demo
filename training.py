import tensorflow as tf
import os
import tensorflow.examples.tutorials.mnist.input_data as input_data

from utils import *
from model import *
from model import BATCH_SIZE

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def train():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_dir = './ckpt'
    images = tf.placeholder(tf.float32, [64, 28, 28, 1], name='real_images')
    z = tf.placeholder(tf.float32, [None, 100], name='z')

    G = generator(z)
    D_logits = discriminator(images)
    samples = sampler(z)
    D_logits_ = discriminator(G, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.losses.sigmoid_cross_entropy(logits=D_logits, multi_class_labels=tf.ones_like(D_logits)))
    d_loss_fake = tf.reduce_mean(
        tf.losses.sigmoid_cross_entropy(logits=D_logits_,multi_class_labels= tf.zeros_like(D_logits_)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(
        tf.losses.sigmoid_cross_entropy(logits=D_logits_, multi_class_labels=tf.ones_like(D_logits_)))

    z_sum = tf.summary.histogram("z", z)
    d_sum = tf.summary.histogram("d", D_logits)
    d__sum = tf.summary.histogram("d_", D_logits_)
    G_sum = tf.summary.histogram("G", G)

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)

    g_sum = tf.summary.merge([z_sum, d__sum, G_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])

    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver = tf.train.Saver()

    d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(g_loss, var_list=g_vars, global_step=global_step)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)

    init = tf.initialize_all_variables()
    writer = tf.summary.FileWriter(train_dir, sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

    sess.run(init)

    for epoch in range(25):
        batch_idxs = 1093
        for idx in range(batch_idxs):
            batch_images = mnist.train.next_batch(batch_size=64)[0]
            batch_images=np.reshape(batch_images,[-1,28,28,1])
            batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))

            _, summary_str,step = sess.run([d_optim, d_sum,global_step],
                                      feed_dict={images: batch_images,
                                                 z: batch_z,})
            writer.add_summary(summary_str, step)

            _, summary_str = sess.run([g_optim, g_sum],
                                      feed_dict={z: batch_z,})
            writer.add_summary(summary_str, step)

            _, summary_str = sess.run([g_optim, g_sum],
                                      feed_dict={z: batch_z,})
            writer.add_summary(summary_str, step)

            errD_fake = d_loss_fake.eval({z: batch_z})
            errD_real = d_loss_real.eval({images: batch_images})
            errG = g_loss.eval({z: batch_z})

            if idx % 20 == 0:
                print("Epoch: [%2d] [%4d/%4d] d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs, errD_fake + errD_real, errG))

            # /home/your_name/TensorFlow/DCGAN/samples/
            if idx % 100 == 1:
                sample = sess.run(samples, feed_dict={z: sample_z})
                samples_path = './images/'
                save_images(sample, [8, 8],
                            samples_path + 'test_%d_epoch_%d.png' % (epoch, idx))
                print 'save down'

            if idx % 500 == 2:
                checkpoint_path = os.path.join(train_dir, 'DCGAN_model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    sess.close()


if __name__ == '__main__':
    train()