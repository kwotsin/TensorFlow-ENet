# TensorFlow-ENet
TensorFlow implementation of [**ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation**](https://arxiv.org/pdf/1606.02147.pdf).

![CamVid Test Dataset Output](https://github.com/kwotsin/TensorFlow-ENet/blob/master/visualizations/original.gif)  ![CamVid Test Dataset Output](https://github.com/kwotsin/TensorFlow-ENet/blob/master/visualizations/output.gif)
### Contents


### Important Notes
As the Max Unpooling layer is not officially available from TensorFlow, a manual implementation was used to build the decoder portion of the network. This was based on the implementation suggested in this [TensorFlow github issue](https://github.com/tensorflow/tensorflow/issues/2169).

Batch normalization and 2D Spatial Dropout are still retained during testing for good performance. 

**Note:** To use the checkpoint model, please set the `stage_two_repeat=3` as the checkpoint was trained on a slightly deeper version of ENet.

### Implementation and Architectural Changes
1. By default, skip connections are added to connect the corresponding encoder and decoder portions for better performance.
2. The number of initial blocks and the depth of stage 2 residual bottlenecks are tunable hyperparameters, to allow you to build a deeper network if required, since ENet is rather lightweight.
3. Fused batch normalization is used over standard batch normalization for faster computations. See [TensorFlow's best practices](https://www.tensorflow.org/performance/performance_guide).
4. To obtain the class weights for computing the weighted loss, Median Frequency Balancing (MFB) is used instead of the custom ENet class weighting function. This is due to an observation that MFB gives a slightly better performance than the custom function, perhaps due to implementation differences. However, the option of obtaining the ENet custom class weights is still possible via the `get_class_weights.py` file.


### References
1. [ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation](https://arxiv.org/pdf/1606.02147.pdf)
2. [Implementation of Max Unpooling](https://github.com/tensorflow/tensorflow/issues/2169)
3. [Implementation of PReLU](https://stackoverflow.com/questions/39975676/how-to-implement-prelu-activation-in-tensorflow)
4. [Clarifications from ENet author](https://github.com/e-lab/ENet-training/issues/56)
