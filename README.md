# TensorFlow-ENet
TensorFlow implementation of [**ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation**](https://arxiv.org/pdf/1606.02147.pdf).

This model was tested on the CamVid dataset with street scenes taken from Cambridge, UK. For more information, please visit: http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/


## Visualizations
Note that the gifs may be out of sync if the network doesn't load them together. You can refresh your page to see them in sync.

### Test Dataset Output
![CamVid Test Dataset Output](https://github.com/kwotsin/TensorFlow-ENet/blob/master/visualizations/original.gif) ![CamVid Test Dataset Output](https://github.com/kwotsin/TensorFlow-ENet/blob/master/visualizations/output.gif)

### TensorBoard Visualizations
Execute `tensorboard --logdir=log` on your root directory to monitor your training and watch your segmentation output form against the ground truth and the original image as you train your model.


## Contents

#### Code
- **enet.py**: The ENet model definition, including the argument scope.

- **train_enet.py**: The file for training. Includes saving of images for visualization and tunable hyperparameters.

- **test_enet.py**: The file for evaluating on the test dataset. Includes option to visualize images as well.

- **preprocessing.py**: The preprocessing does just image resizing, just in case anyone wants to use a smaller image size due                         to memory issues or for other datasets.

- **predict_segmentation.py**: Obtains the segmentation output for visualization purposes. You can create your own gif with                                  these outputs.

- **get_class_weights.py**: The file to obtain either the median frequency balancing class weights, or the custom ENet                                   function class weights.
- **train.sh**: Example training script to train the different variations of the model.

- **test.sh** Example testing script to test the different variants you trained.

#### Folders

- **dataset**: Contains 6 folders that holds the original train-val-test images and their corresponding ground truth annotations.

- **checkpoint**: The checkpoint directory that could be used for predicting the segmentation output. The model was trained using the default parameters mentioned in the paper, except that it uses median frequency balancing to obtain the class weights.
- **visualizations**: Contains the gif files that were created from the output of `predict_segmentation.py`.


## Important Notes
1. As the Max Unpooling layer is not officially available from TensorFlow, a manual implementation was used to build the decoder portion of the network. This was based on the implementation suggested in this [TensorFlow github issue](https://github.com/tensorflow/tensorflow/issues/2169).

2. Batch normalization and 2D Spatial Dropout are still retained during testing for good performance. 

3. Class weights are used to tackle the problem of imbalanced classes, as certain classes appear more dominantly than others. More notably, the background class has weight of 0.0, in order to not reward the model for predicting background.

4. On the labels and colouring scheme: The dataset consists of only 12 labels, with the road-marking class merged with the road class. The last class is the unlabelled class. 

5. No preprocessing is done to the images for ENet. (see references below on clarifications with author), 

## Implementation and Architectural Changes
1. By default, skip connections are added to connect the corresponding encoder and decoder portions for better performance.

2. The number of initial blocks and the depth of stage 2 residual bottlenecks are tunable hyperparameters, to allow you to build a deeper network if required, since ENet is rather lightweight.

3. Fused batch normalization is used over standard batch normalization for faster computations. See [TensorFlow's best practices](https://www.tensorflow.org/performance/performance_guide).

4. To obtain the class weights for computing the weighted loss, Median Frequency Balancing (MFB) is used by default instead of the custom ENet class weighting function. This is due to an observation that MFB gives a slightly better performance than the custom function, at least on my machine. However, the option of using the ENet custom class weights is still possible.

## References
1. [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147.pdf)
2. [Implementation of Max Unpooling](https://github.com/tensorflow/tensorflow/issues/2169)
3. [Implementation of PReLU](https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow)
4. [Clarifications from ENet author](https://github.com/e-lab/ENet-training/issues/56)
5. [Original Torch implementation of ENet](https://github.com/e-lab/ENet-training)
6. [ResNet paper for clarification on residual bottlenecks](https://arxiv.org/pdf/1512.03385.pdf)
7. [Colouring scheme](https://github.com/alexgkendall/SegNet-Tutorial/blob/c922cc4a4fcc7ce279dd998fb2d4a8703f34ebd7/Scripts/test_segmentation_camvid.py)
