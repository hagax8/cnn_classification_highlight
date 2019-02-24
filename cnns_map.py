from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
import cv2

model_path = argv[1]
img_path = argv[2]


def doMod(model_path,img_path,output_name,layer_name):
    model = load_model(model_path)
    last_conv_layer = model.get_layer(layer_name)
    img = image.load_img(img_path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    preds = model.predict(x)
    target_output = model.output
    grads = K.gradients(target_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    nlay = last_conv_layer.get_output_at(0).get_shape().as_list()[-1]
    for i in range(nlay):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    print(preds[[0]])
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    #plt.matshow(heatmap)
    #plt.savefig(output_name+'_heatmap'+'.png')
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap[(img == 255).all(axis=2)]=0
    plt.matshow(heatmap,cmap=plt.cm.cividis)
    plt.savefig(output_name+'_heatmap'+'.png')
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    superimposed_img = heatmap + img
    cv2.imwrite(output_name+'.png', superimposed_img)




def doModArray(model_path,img_path,array,output_path,layer_name):
    model = load_model(model_path)
    last_conv_layer = model.get_layer(layer_name)
    for myimage in array:
        img = image.load_img(img_path+"/"+myimage, target_size=(200, 200))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x /= 255.0
        preds = model.predict(x)
        target_output = model.output
        grads = K.gradients(target_output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input],
                             [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        nlay = last_conv_layer.get_output_at(0).get_shape().as_list()[-1]
        for i in range(nlay):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
        print(preds[[0]])
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        img = cv2.imread(img_path+"/"+myimage)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        #heatmap[(img == 255).all(axis=2)]=0
        plt.matshow(heatmap,cmap=plt.cm.cividis)
        plt.savefig(output_path+"/"+myimage+'_heatmap'+'.png')
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
        superimposed_img = heatmap + img
        cv2.imwrite(output_path+"/"+myimage+'.png', superimposed_img)


















def doMod(model_path,img_path):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(200, 200))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0
    preds = model.predict(x)
    african_elephant_output = model.output
    last_conv_layer = model.get_layer('conv2d_4')
    grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(128):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = -heatmap
    if(np.float(preds[0])>0.5):
        print("INACTIVE")
    else:
        print("ACTIVE")
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    plt.matshow(heatmap)
    import cv2
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_SUMMER)
    superimposed_img = heatmap + img
    cv2.imwrite('test.jpg', superimposed_img)
