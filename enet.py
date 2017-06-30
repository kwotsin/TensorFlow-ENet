import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
slim = tf.contrib.slim

'''
============================================================================
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation
============================================================================
Based on the paper: https://arxiv.org/pdf/1606.02147.pdf
'''
@slim.add_arg_scope
def prelu(x, scope, decoder=False):
    '''
    Performs the parametric relu operation. This implementation is based on:
    https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow

    For the decoder portion, prelu becomes just a normal prelu

    INPUTS:
    - x(Tensor): a 4D Tensor that undergoes prelu
    - scope(str): the string to name your prelu operation's alpha variable.
    - decoder(bool): if True, prelu becomes a normal relu.

    OUTPUTS:
    - pos + neg / x (Tensor): gives prelu output only during training; otherwise, just return x.

    '''
    #If decoder, then perform relu and just return the output
    if decoder:
        return tf.nn.relu(x, name=scope)

    alpha= tf.get_variable(scope + 'alpha', x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alpha * (x - abs(x)) * 0.5
    return pos + neg

def spatial_dropout(x, p, seed, scope, is_training=True):
    '''
    Performs a 2D spatial dropout that drops layers instead of individual elements in an input feature map.
    Note that p stands for the probability of dropping, but tf.nn.relu uses probability of keeping.

    ------------------
    Technical Details
    ------------------
    The noise shape must be of shape [batch_size, 1, 1, num_channels], with the height and width set to 1, because
    it will represent either a 1 or 0 for each layer, and these 1 or 0 integers will be broadcasted to the entire
    dimensions of each layer they interact with such that they can decide whether each layer should be entirely
    'dropped'/set to zero or have its activations entirely kept.
    --------------------------

    INPUTS:
    - x(Tensor): a 4D Tensor of the input feature map.
    - p(float): a float representing the probability of dropping a layer
    - seed(int): an integer for random seeding the random_uniform distribution that runs under tf.nn.relu
    - scope(str): the string name for naming the spatial_dropout
    - is_training(bool): to turn on dropout only when training. Optional.

    OUTPUTS:
    - output(Tensor): a 4D Tensor that is in exactly the same size as the input x,
                      with certain layers having their elements all set to 0 (i.e. dropped).
    '''
    if is_training:
        keep_prob = 1.0 - p
        input_shape = x.get_shape().as_list()
        noise_shape = tf.constant(value=[input_shape[0], 1, 1, input_shape[3]])
        output = tf.nn.dropout(x, keep_prob, noise_shape, seed=seed, name=scope)

        return output

    return x

def unpool(updates, mask, k_size=[1, 2, 2, 1], output_shape=None, scope=''):
    '''
    Unpooling function based on the implementation by Panaetius at https://github.com/tensorflow/tensorflow/issues/2169

    INPUTS:
    - inputs(Tensor): a 4D tensor of shape [batch_size, height, width, num_channels] that represents the input block to be upsampled
    - mask(Tensor): a 4D tensor that represents the argmax values/pooling indices of the previously max-pooled layer
    - k_size(list): a list of values representing the dimensions of the unpooling filter.
    - output_shape(list): a list of values to indicate what the final output shape should be after unpooling
    - scope(str): the string name to name your scope

    OUTPUTS:
    - ret(Tensor): the returned 4D tensor that has the shape of output_shape.

    '''
    with tf.variable_scope(scope):
        mask = tf.cast(mask, tf.int32)
        input_shape = tf.shape(updates, out_type=tf.int32)
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(mask, dtype=tf.int32)
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], 0)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int32), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2] #mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int32)
        f = one_like_mask * feature_range

        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

@slim.add_arg_scope
def initial_block(inputs, is_training=True, scope='initial_block'):
    '''
    The initial block for Enet has 2 branches: The convolution branch and Maxpool branch.

    The conv branch has 13 layers, while the maxpool branch gives 3 layers corresponding to the RGB channels.
    Both output layers are then concatenated to give an output of 16 layers.

    NOTE: Does not need to store pooling indices since it won't be used later for the final upsampling.

    INPUTS:
    - inputs(Tensor): A 4D tensor of shape [batch_size, height, width, channels]

    OUTPUTS:
    - net_concatenated(Tensor): a 4D Tensor that contains the 
    '''
    #Convolutional branch
    net_conv = slim.conv2d(inputs, 13, [3,3], stride=2, activation_fn=None, scope=scope+'_conv')
    net_conv = slim.batch_norm(net_conv, is_training=is_training, fused=True, scope=scope+'_batchnorm')
    net_conv = prelu(net_conv, scope=scope+'_prelu')

    #Max pool branch
    net_pool = slim.max_pool2d(inputs, [2,2], stride=2, scope=scope+'_max_pool')

    #Concatenated output - does it matter max pool comes first or conv comes first? probably not.
    net_concatenated = tf.concat([net_conv, net_pool], axis=3, name=scope+'_concat')
    return net_concatenated

@slim.add_arg_scope
def bottleneck(inputs,
               output_depth,
               filter_size,
               regularizer_prob,
               projection_ratio=4,
               seed=0,
               is_training=True,
               downsampling=False,
               upsampling=False,
               pooling_indices=None,
               output_shape=None,
               dilated=False,
               dilation_rate=None,
               asymmetric=False,
               decoder=False,
               scope='bottleneck'):
    '''
    The bottleneck module has three different kinds of variants:

    1. A regular convolution which you can decide whether or not to downsample.
    2. A dilated convolution, which requires you to have a dilation factor.
    3. An asymmetric convolution that has a decomposed filter size of 5x1 and 1x5 separately.

    INPUTS:
    - inputs(Tensor): a 4D Tensor of the previous convolutional block of shape [batch_size, height, width, num_channels].
    - output_depth(int): an integer indicating the output depth of the output convolutional block.
    - filter_size(int): an integer that gives the height and width of the filter size to use for a regular/dilated convolution.
    - regularizer_prob(float): the float p that represents the prob of dropping a layer for spatial dropout regularization.
    - projection_ratio(int): the amount of depth to reduce for initial 1x1 projection. Depth is divided by projection ratio. Default is 4.
    - seed(int): an integer for the random seed used in the random normal distribution within dropout.
    - is_training(bool): a boolean value to indicate whether or not is training. Decides batch_norm and prelu activity.

    - downsampling(bool): if True, a max-pool2D layer is added to downsample the spatial sizes.
    - upsampling(bool): if True, the upsampling bottleneck is activated but requires pooling indices to upsample.
    - pooling_indices(Tensor): the argmax values that are obtained after performing tf.nn.max_pool_with_argmax.
    - output_shape(list): A list of integers indicating the output shape of the unpooling layer.
    - dilated(bool): if True, then dilated convolution is done, but requires a dilation rate to be given.
    - dilation_rate(int): the dilation factor for performing atrous convolution/dilated convolution.
    - asymmetric(bool): if True, then asymmetric convolution is done, and the only filter size used here is 5.
    - decoder(bool): if True, then all the prelus become relus according to ENet author.
    - scope(str): a string name that names your bottleneck.

    OUTPUTS:
    - net(Tensor): The convolution block output after a bottleneck
    - pooling_indices(Tensor): If downsample, then this tensor is produced for use in upooling later.
    - inputs_shape(list): The shape of the input to the downsampling conv block. For use in unpooling later.

    '''
    #Calculate the depth reduction based on the projection ratio used in 1x1 convolution.
    reduced_depth = int(inputs.get_shape().as_list()[3] / projection_ratio)

    with slim.arg_scope([prelu], decoder=decoder):

        #=============DOWNSAMPLING BOTTLENECK====================
        if downsampling:
            #=============MAIN BRANCH=============
            #Just perform a max pooling
            net_main, pooling_indices = tf.nn.max_pool_with_argmax(inputs,
                                                                   ksize=[1,2,2,1],
                                                                   strides=[1,2,2,1],
                                                                   padding='SAME',
                                                                   name=scope+'_main_max_pool')

            #First get the difference in depth to pad, then pad with zeros only on the last dimension.
            inputs_shape = inputs.get_shape().as_list()
            depth_to_pad = abs(inputs_shape[3] - output_depth)
            paddings = tf.convert_to_tensor([[0,0], [0,0], [0,0], [0, depth_to_pad]])
            net_main = tf.pad(net_main, paddings=paddings, name=scope+'_main_padding')

            #=============SUB BRANCH==============
            #First projection that has a 2x2 kernel and stride 2
            net = slim.conv2d(inputs, reduced_depth, [2,2], stride=2, scope=scope+'_conv1')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm1')
            net = prelu(net, scope=scope+'_prelu1')

            #Second conv block
            net = slim.conv2d(net, reduced_depth, [filter_size, filter_size], scope=scope+'_conv2')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
            net = prelu(net, scope=scope+'_prelu2')

            #Final projection with 1x1 kernel
            net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
            net = prelu(net, scope=scope+'_prelu3')

            #Regularizer
            net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')

            #Finally, combine the two branches together via an element-wise addition
            net = tf.add(net, net_main, name=scope+'_add')
            net = prelu(net, scope=scope+'_last_prelu')

            #also return inputs shape for convenience later
            return net, pooling_indices, inputs_shape

        #============DILATION CONVOLUTION BOTTLENECK====================
        #Everything is the same as a regular bottleneck except for the dilation rate argument
        elif dilated:
            #Check if dilation rate is given
            if not dilation_rate:
                raise ValueError('Dilation rate is not given.')

            #Save the main branch for addition later
            net_main = inputs

            #First projection with 1x1 kernel (dimensionality reduction)
            net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm1')
            net = prelu(net, scope=scope+'_prelu1')

            #Second conv block --- apply dilated convolution here
            net = slim.conv2d(net, reduced_depth, [filter_size, filter_size], rate=dilation_rate, scope=scope+'_dilated_conv2')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
            net = prelu(net, scope=scope+'_prelu2')

            #Final projection with 1x1 kernel (Expansion)
            net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
            net = prelu(net, scope=scope+'_prelu3')

            #Regularizer
            net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')
            net = prelu(net, scope=scope+'_prelu4')

            #Add the main branch
            net = tf.add(net_main, net, name=scope+'_add_dilated')
            net = prelu(net, scope=scope+'_last_prelu')

            return net

        #===========ASYMMETRIC CONVOLUTION BOTTLENECK==============
        #Everything is the same as a regular bottleneck except for a [5,5] kernel decomposed into two [5,1] then [1,5]
        elif asymmetric:
            #Save the main branch for addition later
            net_main = inputs

            #First projection with 1x1 kernel (dimensionality reduction)
            net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm1')
            net = prelu(net, scope=scope+'_prelu1')

            #Second conv block --- apply asymmetric conv here
            net = slim.conv2d(net, reduced_depth, [filter_size, 1], scope=scope+'_asymmetric_conv2a')
            net = slim.conv2d(net, reduced_depth, [1, filter_size], scope=scope+'_asymmetric_conv2b')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
            net = prelu(net, scope=scope+'_prelu2')

            #Final projection with 1x1 kernel
            net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
            net = prelu(net, scope=scope+'_prelu3')

            #Regularizer
            net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')
            net = prelu(net, scope=scope+'_prelu4')

            #Add the main branch
            net = tf.add(net_main, net, name=scope+'_add_asymmetric')
            net = prelu(net, scope=scope+'_last_prelu')

            return net

        #============UPSAMPLING BOTTLENECK================
        #Everything is the same as a regular one, except convolution becomes transposed.
        elif upsampling:
            #Check if pooling indices is given
            if pooling_indices == None:
                raise ValueError('Pooling indices are not given.')

            #Check output_shape given or not
            if output_shape == None:
                raise ValueError('Output depth is not given')

            #=======MAIN BRANCH=======
            #Main branch to upsample. output shape must match with the shape of the layer that was pooled initially, in order
            #for the pooling indices to work correctly. However, the initial pooled layer was padded, so need to reduce dimension
            #before unpooling. In the paper, padding is replaced with convolution for this purpose of reducing the depth!
            net_unpool = slim.conv2d(inputs, output_depth, [1,1], scope=scope+'_main_conv1')
            net_unpool = slim.batch_norm(net_unpool, is_training=is_training, scope=scope+'batch_norm1')
            net_unpool = unpool(net_unpool, pooling_indices, output_shape=output_shape, scope='unpool')

            #======SUB BRANCH=======
            #First 1x1 projection to reduce depth
            net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
            net = prelu(net, scope=scope+'_prelu1')

            #Second conv block -----------------------------> NOTE: using tf.nn.conv2d_transpose for variable input shape.
            net_unpool_shape = net_unpool.get_shape().as_list()
            output_shape = [net_unpool_shape[0], net_unpool_shape[1], net_unpool_shape[2], reduced_depth]
            output_shape = tf.convert_to_tensor(output_shape)
            filter_size = [filter_size, filter_size, reduced_depth, reduced_depth]
            filters = tf.get_variable(shape=filter_size, initializer=initializers.xavier_initializer(), dtype=tf.float32, name=scope+'_transposed_conv2_filters')

            # net = slim.conv2d_transpose(net, reduced_depth, [filter_size, filter_size], stride=2, scope=scope+'_transposed_conv2')
            net = tf.nn.conv2d_transpose(net, filter=filters, strides=[1,2,2,1], output_shape=output_shape, name=scope+'_transposed_conv2')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
            net = prelu(net, scope=scope+'_prelu2')

            #Final projection with 1x1 kernel
            net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
            net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm4')
            net = prelu(net, scope=scope+'_prelu3')

            #Regularizer
            net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')
            net = prelu(net, scope=scope+'_prelu4')

            #Finally, add the unpooling layer and the sub branch together
            net = tf.add(net, net_unpool, name=scope+'_add_upsample')
            net = prelu(net, scope=scope+'_last_prelu')

            return net

        #OTHERWISE, just perform a regular bottleneck!
        #==============REGULAR BOTTLENECK==================
        #Save the main branch for addition later
        net_main = inputs

        #First projection with 1x1 kernel
        net = slim.conv2d(inputs, reduced_depth, [1,1], scope=scope+'_conv1')
        net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm1')
        net = prelu(net, scope=scope+'_prelu1')

        #Second conv block
        net = slim.conv2d(net, reduced_depth, [filter_size, filter_size], scope=scope+'_conv2')
        net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm2')
        net = prelu(net, scope=scope+'_prelu2')

        #Final projection with 1x1 kernel
        net = slim.conv2d(net, output_depth, [1,1], scope=scope+'_conv3')
        net = slim.batch_norm(net, is_training=is_training, scope=scope+'_batch_norm3')
        net = prelu(net, scope=scope+'_prelu3')

        #Regularizer
        net = spatial_dropout(net, p=regularizer_prob, seed=seed, scope=scope+'_spatial_dropout')
        net = prelu(net, scope=scope+'_prelu4')

        #Add the main branch
        net = tf.add(net_main, net, name=scope+'_add_regular')
        net = prelu(net, scope=scope+'_last_prelu')

        return net

#Now actually start building the network
def ENet(inputs,
         num_classes,
         batch_size,
         num_initial_blocks=1,
         stage_two_repeat=2,
         skip_connections=True,
         reuse=None,
         is_training=True,
         scope='ENet'):
    '''
    The ENet model for real-time semantic segmentation!

    INPUTS:
    - inputs(Tensor): a 4D Tensor of shape [batch_size, image_height, image_width, num_channels] that represents one batch of preprocessed images.
    - num_classes(int): an integer for the number of classes to predict. This will determine the final output channels as the answer.
    - batch_size(int): the batch size to explictly set the shape of the inputs in order for operations to work properly.
    - num_initial_blocks(int): the number of times to repeat the initial block.
    - stage_two_repeat(int): the number of times to repeat stage two in order to make the network deeper.
    - skip_connections(bool): if True, add the corresponding encoder feature maps to the decoder. They are of exact same shapes.
    - reuse(bool): Whether or not to reuse the variables for evaluation.
    - is_training(bool): if True, switch on batch_norm and prelu only during training, otherwise they are turned off.
    - scope(str): a string that represents the scope name for the variables.

    OUTPUTS:
    - net(Tensor): a 4D Tensor output of shape [batch_size, image_height, image_width, num_classes], where each pixel has a one-hot encoded vector
                      determining the label of the pixel.
    '''
    #Set the shape of the inputs first to get the batch_size information
    inputs_shape = inputs.get_shape().as_list()
    inputs.set_shape(shape=(batch_size, inputs_shape[1], inputs_shape[2], inputs_shape[3]))

    with tf.variable_scope(scope, reuse=reuse):
        #Set the primary arg scopes. Fused batch_norm is faster than normal batch norm.
        with slim.arg_scope([initial_block, bottleneck], is_training=is_training),\
             slim.arg_scope([slim.batch_norm], fused=True), \
             slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=None): 
            #=================INITIAL BLOCK=================
            for i in xrange(1, max(num_initial_blocks, 1) + 1):
                net = initial_block(inputs, scope='initial_block_' + str(i))

            #Save for skip connection later
            if skip_connections:
                net_one = net

            #===================STAGE ONE=======================
            net, pooling_indices_1, inputs_shape_1 = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, downsampling=True, scope='bottleneck1_0')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_1')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_2')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_3')
            net = bottleneck(net, output_depth=64, filter_size=3, regularizer_prob=0.01, scope='bottleneck1_4')

            #Save for skip connection later
            if skip_connections:
                net_two = net

            #regularization prob is 0.1 from bottleneck 2.0 onwards
            with slim.arg_scope([bottleneck], regularizer_prob=0.1):
                net, pooling_indices_2, inputs_shape_2 = bottleneck(net, output_depth=128, filter_size=3, downsampling=True, scope='bottleneck2_0')
                
                #Repeat the stage two at least twice to get stage 2 and 3:
                for i in xrange(2, max(stage_two_repeat, 2) + 2):
                    net = bottleneck(net, output_depth=128, filter_size=3, scope='bottleneck'+str(i)+'_1')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=2, scope='bottleneck'+str(i)+'_2')
                    net = bottleneck(net, output_depth=128, filter_size=5, asymmetric=True, scope='bottleneck'+str(i)+'_3')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=4, scope='bottleneck'+str(i)+'_4')
                    net = bottleneck(net, output_depth=128, filter_size=3, scope='bottleneck'+str(i)+'_5')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=8, scope='bottleneck'+str(i)+'_6')
                    net = bottleneck(net, output_depth=128, filter_size=5, asymmetric=True, scope='bottleneck'+str(i)+'_7')
                    net = bottleneck(net, output_depth=128, filter_size=3, dilated=True, dilation_rate=16, scope='bottleneck'+str(i)+'_8')

            with slim.arg_scope([bottleneck], regularizer_prob=0.1, decoder=True):
                #===================STAGE FOUR========================
                bottleneck_scope_name = "bottleneck" + str(i + 1)

                #The decoder section, so start to upsample.
                net = bottleneck(net, output_depth=64, filter_size=3, upsampling=True,
                                 pooling_indices=pooling_indices_2, output_shape=inputs_shape_2, scope=bottleneck_scope_name+'_0')

                #Perform skip connections here
                if skip_connections:
                    net = tf.add(net, net_two, name=bottleneck_scope_name+'_skip_connection')

                net = bottleneck(net, output_depth=64, filter_size=3, scope=bottleneck_scope_name+'_1')
                net = bottleneck(net, output_depth=64, filter_size=3, scope=bottleneck_scope_name+'_2')

                #===================STAGE FIVE========================
                bottleneck_scope_name = "bottleneck" + str(i + 2)

                net = bottleneck(net, output_depth=16, filter_size=3, upsampling=True,
                                 pooling_indices=pooling_indices_1, output_shape=inputs_shape_1, scope=bottleneck_scope_name+'_0')

                #perform skip connections here
                if skip_connections:
                    net = tf.add(net, net_one, name=bottleneck_scope_name+'_skip_connection')

                net = bottleneck(net, output_depth=16, filter_size=3, scope=bottleneck_scope_name+'_1')

            #=============FINAL CONVOLUTION=============
            logits = slim.conv2d_transpose(net, num_classes, [2,2], stride=2, scope='fullconv')
            probabilities = tf.nn.softmax(logits, name='logits_to_softmax')

        return logits, probabilities


def ENet_arg_scope(weight_decay=2e-4,
                   batch_norm_decay=0.1,
                   batch_norm_epsilon=0.001):
  '''
  The arg scope for enet model. The weight decay is 2e-4 as seen in the paper.
  Batch_norm decay is 0.1 (momentum 0.1) according to official implementation.

  INPUTS:
  - weight_decay(float): the weight decay for weights variables in conv2d and separable conv2d
  - batch_norm_decay(float): decay for the moving average of batch_norm momentums.
  - batch_norm_epsilon(float): small float added to variance to avoid dividing by zero.

  OUTPUTS:
  - scope(arg_scope): a tf-slim arg_scope with the parameters needed for xception.
  '''
  # Set weight_decay for weights in conv2d and separable_conv2d layers.
  with slim.arg_scope([slim.conv2d],
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_regularizer=slim.l2_regularizer(weight_decay)):

    # Set parameters for batch_norm.
    with slim.arg_scope([slim.batch_norm],
                        decay=batch_norm_decay,
                        epsilon=batch_norm_epsilon) as scope:
      return scope