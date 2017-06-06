# Needs Python 2.7 and Keras 1.2.2
#
# To set this up:
# virtualenv -p /usr/bin/python2.7 env
# source env/bin/activate
# pip install tensorflow
# pip install keras==1.2.2
# pip install coremltools
#
# Use "deactivate" when you're done.

import coremltools

# Note: It appears that coremltools applies the scale *before* subtracting
# the means.
scale = 0.017

coreml_model = coremltools.converters.caffe.convert(
    ('mobilenet.caffemodel', 'mobilenet_deploy.prototxt'),
    image_input_names='data',
    is_bgr=True, image_scale=scale,
    red_bias=123.68*scale, green_bias=116.78*scale, blue_bias=103.94*scale,
    class_labels='synset_words.txt')

coreml_model.author = 'Original Paper: Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. Caffe implementation: shicai'
coreml_model.license = 'Unknown'
coreml_model.short_description = "The network from the paper 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications', trained on the ImageNet dataset."

coreml_model.input_description['data'] = 'Input image to be classified'

coreml_model.output_description['prob'] = 'Probability of each category'
coreml_model.output_description['classLabel'] = 'Most likely image category'

coreml_model.save('MobileNet.mlmodel')
