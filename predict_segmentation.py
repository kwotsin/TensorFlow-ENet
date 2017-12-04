import tensorflow as tf
import os
import matplotlib.pyplot as plt
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess
from scipy.misc import imsave
import numpy as np
slim = tf.contrib.slim


#==============INPUT ARGUMENTS==================
flags = tf.app.flags

#Parameters
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 10, 'The batch_size for training.')

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')

#Directory arguments
flags.DEFINE_string('checkpoint_dir', './checkpoint_mfb', 'The checkpoint directory to restore your model.')
flags.DEFINE_string('image_dir', './dataset/test', 'The image directory to find the images to predict.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size

#Architectural changes
num_initial_blocks = FLAGS.num_initial_blocks
skip_connections = FLAGS.skip_connections
stage_two_repeat = FLAGS.stage_two_repeat

#Directories
image_dir = FLAGS.image_dir
checkpoint_dir = FLAGS.checkpoint_dir

#Directories
#Get the list of images to predict in image_dir
images_list = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.png')])
#Get latest checkpoint in checkpoint_dir
checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
#Create the photo directory
photo_dir = checkpoint_dir + "/test_images"
if not os.path.exists(photo_dir):
    os.mkdir(photo_dir)

'''
#Labels to colours are obtained from here:
https://github.com/alexgkendall/SegNet-Tutorial/blob/c922cc4a4fcc7ce279dd998fb2d4a8703f34ebd7/Scripts/test_segmentation_camvid.py

However, the road_marking class is collapsed into the road class in the dataset provided.

Classes:
------------
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
'''
label_to_colours =    {0: [128,128,128],
                     1: [128,0,0],
                     2: [192,192,128],
                     3: [128,64,128],
                     4: [60,40,222],
                     5: [128,128,0],
                     6: [192,128,128],
                     7: [64,64,128],
                     8: [64,0,128],
                     9: [64,64,0],
                     10: [0,128,192],
                     11: [0,0,0]}

#Create a function to convert each pixel label to colour.
def grayscale_to_colour(image):
    print 'Converting image...'
    image = image.reshape((image.shape[0], image.shape[1], 1))
    image = np.repeat(image, 3, axis=-1)
    for i in xrange(image.shape[0]):
        for j in xrange(image.shape[1]):
            label = int(image[i][j][0])
            image[i][j] = np.array(label_to_colours[label])

    return image


with tf.Graph().as_default() as graph:
    images_tensor = tf.train.string_input_producer(images_list, shuffle=False)
    reader = tf.WholeFileReader()
    key, image_tensor = reader.read(images_tensor)
    image = tf.image.decode_png(image_tensor, channels=3)
    # image = tf.image.resize_image_with_crop_or_pad(image, 360, 480)
    # image = tf.cast(image, tf.float32)
    image = preprocess(image)
    images = tf.train.batch([image], batch_size = batch_size, allow_smaller_final_batch=True)

    #Create the model inference
    with slim.arg_scope(ENet_arg_scope()):
        logits, probabilities = ENet(images,
                                     num_classes=num_classes,
                                     batch_size=batch_size,
                                     is_training=True,
                                     reuse=None,
                                     num_initial_blocks=num_initial_blocks,
                                     stage_two_repeat=stage_two_repeat,
                                     skip_connections=skip_connections)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint)

    predictions = tf.argmax(probabilities, -1)
    predictions = tf.cast(predictions, tf.float32)
    print 'HERE', predictions.get_shape()

    sv = tf.train.Supervisor(logdir=None, init_fn=restore_fn)

    with sv.managed_session() as sess:

        for i in xrange(len(images_list) / batch_size + 1):
            segmentations = sess.run(predictions)
            # print segmentations.shape

            for j in xrange(segmentations.shape[0]):
                current_image = i * batch_size + j

                if current_image > len(images_list) - 1:
                    break

                converted_image = grayscale_to_colour(segmentations[j])
                print 'Saving image %s/%s' %(current_image, len(images_list) - 1)
                plt.axis('off')
                plt.imshow(converted_image)
                imsave(photo_dir + "/image_%s.png" %(current_image), converted_image)
                # plt.show()
