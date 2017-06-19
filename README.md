# Implementations of Neural Style Transfer in TF/Keras

![alt text](https://github.com/xuberance137/styleflow/blob/master/remote/base5_ref6__at_iteration_01.png)

neural_style contains a perliminary neural style algorithm adapted from https://github.com/anishathalye/neural-style

Example Run:
python neural_style.py --content ./images/input.png --styles ./images/style.jpg --output ./images/output.jpg --iterations 10

neural_transfer implements the algorithm in reference [1] with Keras, adapted from the Keras examples

Changing nature of content loss and style loss from original implementation

Run the script with:
python neural_style_transfer.py path_to_your_base_image.jpg path_to_your_reference.jpg prefix_for_results

Example Run:
python neural_transfer_02.py ~/Downloads/base1.jpg ~/Downloads/ref1.jpg neural_test_

Optional parameters:
--iter, To specify the number of iterations the style transfer takes place (Default is 10)
--content_weight, The weight given to the content loss (Default is 0.025)
--style_weight, The weight given to the style loss (Default is 1.0)
--tv_weight, The weight given to the total variation loss (Default is 1.0)

# Details

Style transfer consists in generating an image with the same "content" as a base image, but with the "style" of a different picture (typically artistic).

Starts with the VGG19 model without top/FC layers. 
Content loss is derived from higher level features - block5_conv2 - of the base image.
Style loss is based on lower level features - block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1' - of the reference image.
To transfer the style of an artwork a(ref image) onto a photograph p (base image), we synthesize a new image that 
simultaneously matches the content representation of p and the style representation of a.

This is achieved through the optimization of a loss function that has 3 components: "style loss", "content loss" and "total variation loss":

- The total variation loss imposes local spatial continuity between
the pixels of the combination image, giving it visual coherence. It looks at the L2 norm of the sub image with when shifted by a single pixel to the left and down.

- The style loss is where the deep learning keeps in --that one is defined
using a deep convolutional neural network. Precisely, it consists in a sum of
L2 distances between the Gram matrices of the representations of
the base image and the style reference image, extracted from
different layers of a convnet (trained on ImageNet). The general idea
is to capture color/texture information at different spatial
scales (fairly large scales --defined by the depth of the layer considered).

 - The content loss is a L2 distance between the features of the base
image (extracted from a deep layer) and the features of the combination image,
keeping the generated image close enough to the original one.

Each iteration on the combination image is an iteration on the nonlinear optimizer to get the lowest loss given the input base image, the loss function
and the derivative/gradient (wrt to the combination image). Uses the scipy implementation of the BFGS optimizer. Iteration time depends on the size of the target image chosen. For a target width of 400, it takes ~20 seconds to run a single iteration on a p2 instance with the K80/GPU and ~300 seconds to run on i7/CPU. When raising the traget width to 800, it takes ~1500 seconds to run on i7/CPU

# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
    - [Deep Photo Style Transfer](https://arxiv.org/abs/1703.07511)
    - https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
    - https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
