Neural style transfer with Keras, adapted from 
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

Changing nature of content loss and style loss from original implementation

Run the script with:
python neural_style_transfer.py path_to_your_base_image.jpg path_to_your_reference.jpg prefix_for_results

e.g.:
python neural_transfer_02.py ~/Downloads/base1.jpg ~/Downloads/ref1.jpg neural_test_

Optional parameters:
--iter, To specify the number of iterations the style transfer takes place (Default is 10)
--content_weight, The weight given to the content loss (Default is 0.025)
--style_weight, The weight given to the style loss (Default is 1.0)
--tv_weight, The weight given to the total variation loss (Default is 1.0)

# Details

Style transfer consists in generating an image with the same "content" as a base image, but with the "style" of a different picture (typically artistic).

Starts with the VGG19 model without top/FC layers. 
Content loss is derived from higher level features - block5_conv2 - of the base image
Style loss is based on lower level features - block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1' - of the reference image
To transfer the style of an artwork a(ref image) onto a photograph p (base image), we synthesize a new image that 
simultaneously matches the content representation of p and the style representation of a.

This is achieved through the optimization of a loss function
that has 3 components: "style loss", "content loss",
and "total variation loss":

- The total variation loss imposes local spatial continuity between
the pixels of the combination image, giving it visual coherence.

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

# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)