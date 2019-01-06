'''

Run the script with:
python deep_dream_viz.py

e.g.:
python deep_dream_viz.py


'''

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.applications.densenet import DenseNet121, preprocess_input, decode_predictions
from keras.applications import inception_v3
from keras.preprocessing import image
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy
import imageio

TEST_MODEL = False
IMG_SIZE = 100
EPSILON = 1e-7
NUM_ITERATIONS = 10
INTERVAL_SIZE = NUM_ITERATIONS // 10 # this needs to be set as 1/10th of the number of iterations
NUM_FILTERS = 5
NUM_INTERVALS = 10 

'''
command line arguments
'''
def parse_and_set_arguments():
	global layer_name 	#layer_name = 'res3a_branch2a' #'block3_conv1'
	global initial_filter_id
	parser = argparse.ArgumentParser(description='Filter visualization with gradient ascent')
	parser.add_argument('layer_name', metavar='base', type=str, help='Layer Name for visualization')
	parser.add_argument('initial_filter_id', metavar='ref', type=int, help='Initial filter amongst 10 for visualization')
	args = parser.parse_args()
	# set values
	layer_name = args.layer_name
	initial_filter_id = args.initial_filter_id


def resize_img(img, size):
	img = np.copy(img)
	factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1)
	return scipy.ndimage.zoom(img, factors, order=1)

def save_img(img, fname):
	pil_img = deprocess_image(np.copy(img))
	#scipy.misc.imsave(fname, pil_img)
	imageio.imwrite(fname, pil_img)


def preprocess_image(image_path):
	# Util function to open, resize and format picture into appropriate tensors.
	img = image.load_img(image_path)
	img = image.img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = inception_v3.preprocess_input(img)
	return img

'''
Going from unbounded tensor to scaled tensor, scales and clipped to 8 bit integer numpy array, represented as an image
'''
def deprocess_image(x):
	# Util function to convert a tensor into a valid image.
	if K.image_data_format() == 'channels_first':
		x = x.reshape((3, x.shape[2], x.shape[3]))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((x.shape[1], x.shape[2], 3))
	x /= 2.
	x += 0.5
	x *= 255.
	x = np.clip(x, 0, 255).astype('uint8')
	return x



'''
We can get to filter visualizations with gradient ascent in input space: applying gradient 
descent to the value of the input image of a convnet so as to maximize the response of a 
specific filter, starting from a blank input image. The resulting input image would 
be one that the chosen filter is maximally responsive to.

The process is simple: we will build a loss function that maximizes the value of a 
given filter in a given convolution layer, then we will use stochastic gradient 
descent to adjust the values of the input image so as to maximize this activation value.
'''
def construct_loss(model, layer_contributions):

	layer_dict = dict([layer.name, layer] for layer in model.layers)

	loss = K.variable(0.0)

	for layer_name in layer_contributions:
		coeff = layer_contributions[layer_name]
		activation = layer_dict[layer_name]

		# mean of sum of squares (L2norm) of activations scaled by contribution constitutes the loss.
		# avoiding border artifacts 
		scaling = K.prod(K.cast(K.shape(activation), 'float32'))
		loss += coeff*K.sum(K.square(activation[:,2:-2, 2:-2, :]))/scaling

	dream = model.input

	# The call to `gradients` returns a list of tensors (of size 1 in this case)
	# hence we only keep the first element -- which is a tensor.
	grads = K.gradients(loss, dream)[0]
	 # Normalized gradients. We floor with 1e-7 before dividing so as to avoid accidentally dividing by 0.
	# grads /= (K.sqrt(K.mean(K.square(grads))) + EPSILON)
	grads /= K.maximum(K.mean(L.abs(grads)), EPSILON)

	# going from input image to loss and gradients
	iterate = K.function([dream], [loss, grads])

	return iterate

def gradient_ascent(x, generate_gradients, iterations, step, max_loss=None):
	for i in range(iterations):
		# Compute the loss value and gradient value
		[loss_value, grads_value] = generate_gradients([x])
		if max_loss is not None and loss_value > max_loss:
			break
		print('Loss Value at iteration ', i, ' : ', loss_value)
		# Here we adjust the input image in the direction that maximizes the loss
		x += grads_value * step

	return x

def generate_dream(img, base_image_name, model, layer_contributions, iterations, num_octave, octave_scale, gradient_step, max_loss):
	
	#construct loss
	layer_dict = dict([layer.name, layer] for layer in model.layers)

	loss = K.variable(0.0)

	for layer_name in layer_contributions:
		coeff = layer_contributions[layer_name]
		activation = layer_dict[layer_name].output
		print(coeff, activation, K.shape(activation))

	for layer_name in layer_contributions:
		coeff = layer_contributions[layer_name]
		activation = layer_dict[layer_name].output

		# mean of sum of squares (L2norm) of activations scaled by contribution constitutes the loss.
		# avoiding border artifacts 
		scaling = K.prod(K.cast(K.shape(activation), 'float32'))
		loss += coeff*K.sum(K.square(activation[:,2:-2, 2:-2, :]))/scaling

	dream = model.input

	# The call to `gradients` returns a list of tensors (of size 1 in this case)
	# hence we only keep the first element -- which is a tensor.
	grads = K.gradients(loss, dream)[0]
	 # Normalized gradients. We floor with 1e-7 before dividing so as to avoid accidentally dividing by 0.
	# grads /= (K.sqrt(K.mean(K.square(grads))) + EPSILON)
	grads /= K.maximum(K.mean(K.abs(grads)), EPSILON)
	# going from input image to loss and gradients
	generate_gradients = K.function([dream], [loss, grads])

	#generate scaled variants
	original_shape = img.shape[1:3]
	successive_shapes = [original_shape]
	for i in range(1, num_octave):
		shape  = tuple([int(dim/(octave_scale **i)) for dim in original_shape])
		successive_shapes.append(shape)
	#reverse shapes to be in ascending order
	successive_shapes = successive_shapes[::-1]

	#original full scale image
	original_img = np.copy(img)
	# set shrunk image to smallest scale
	shrunk_original_img = resize_img(img, successive_shapes[0])

	for shape in successive_shapes:
		print('Processing image for shape : ', shape)
		img = resize_img(img, shape)
		img = gradient_ascent(img, generate_gradients, iterations=iterations, step=gradient_step, max_loss=max_loss)
		upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
		same_size_original = resize_img(original_img, shape)
		lost_detail = same_size_original - upscaled_shrunk_original_img
		img += lost_detail
		shrunk_original_img = resize_img(original_img, shape)
		save_img(img, fname='./dreams/'+base_image_name.split('/')[-1][:-4]+'_dream_at_scale_'+str(shape)+'.png')

	save_img(img, fname='./dreams/'+base_image_name.split('/')[-1][:-4]+'_final_dream.png')


'''
Main Function
'''
if __name__ == '__main__':

	#parse_and_set_arguments()

	# coefficients indicate how much the layers contributed to the loss that we aim to maximize
	layer_contributions = {
		'mixed2': 0.2,
		'mixed3': 3.0,
		'mixed4': 2.0,
		'mixed5': 1.5,
	}

	gradient_step = 0.01
	num_octave = 7 # scales at which gradient ascent is run
	octave_scale = 1.4 # ratio between scales (sq rt of 2)
	iterations = 100 #number of steps
	max_loss = 30.0 #either max out the number of iterations or hit the max loss and break
	base_image_name = './images/IMG_2205.jpg'

	# disable training specific operations
	K.set_learning_phase(0)
	model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

	for i, l in enumerate(model.layers):
		print(i, l.name, l.output_shape)

	if TEST_MODEL:
		img_path = '/Users/gopal/Downloads/base5.jpg'
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		preds = model.predict(x)
		print('Predicted:', decode_predictions(preds, top=5)[0])

	else:
		img = preprocess_image(base_image_name)
		dream = generate_dream(img, base_image_name, model, layer_contributions, iterations, num_octave, octave_scale, gradient_step, max_loss)


