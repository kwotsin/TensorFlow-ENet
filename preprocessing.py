import tensorflow as tf

def preprocess(image, annotation=None, height=360, width=480):
    '''
    Performs preprocessing for one set of image and annotation for feeding into network.
    NO scaling of any sort will be done as per original paper.

    INPUTS:
    - image (Tensor): the image input 3D Tensor of shape [height, width, 3]
    - annotation (Tensor): the annotation input 3D Tensor of shape [height, width, 1]
    - height (int): the output height to reshape the image and annotation into
    - width (int): the output width to reshape the image and annotation into

    OUTPUTS:
    - preprocessed_image(Tensor): the reshaped image tensor
    - preprocessed_annotation(Tensor): the reshaped annotation tensor
    '''

    #Convert the image and annotation dtypes to tf.float32 if needed
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # image = tf.cast(image, tf.float32)

    image = tf.image.resize_image_with_crop_or_pad(image, height, width)
    image.set_shape(shape=(height, width, 3))

    if not annotation == None:
        annotation = tf.image.resize_image_with_crop_or_pad(annotation, height, width)
        annotation.set_shape(shape=(height, width, 1))

        return image, annotation

    return image