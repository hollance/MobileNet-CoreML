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
# the means. So we have to scale the mean RGB by this factor too.
scale = 0.017

coreml_model = coremltools.converters.caffe.convert(
    ('mobilenet.caffemodel', 'mobilenet_deploy.prototxt'),
    image_input_names='data',
    is_bgr=True, image_scale=scale,
    red_bias=-123.68*scale, green_bias=-116.78*scale, blue_bias=-103.94*scale,
    class_labels='synset_words.txt')

coreml_model.author = 'Original paper: Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. Caffe implementation: shicai'
coreml_model.license = 'Unknown'
coreml_model.short_description = "The network from the paper 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications', trained on the ImageNet dataset."

coreml_model.input_description['data'] = 'Input image to be classified'

coreml_model.output_description['prob'] = 'Probability of each category'
coreml_model.output_description['classLabel'] = 'Most likely image category'

print(coreml_model)

# Test that the converted network gives the same output as the original
# model. The top 5 predictions should be:
#   0.29618 n02123159 tiger cat 282
#   0.14749 n02119022 red fox, Vulpes vulpes 277
#   0.13466 n02119789 kit fox, Vulpes macrotis 278
#   0.08651 n02113023 Pembroke, Pembroke Welsh corgi 263
#   0.03148 n02123045 tabby, tabby cat 281
#
# To run this you need macOS 10.13 and the following packages:
#   pip install pillow

#from PIL import Image
#cat = Image.open('../cat.jpg')
#print(coreml_model.predict({'data': cat}))

coreml_model.save('MobileNet.mlmodel')
