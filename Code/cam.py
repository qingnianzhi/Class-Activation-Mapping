#coding:utf-8

import caffe
import numpy as np
import cv2
import matplotlib.pylab as plt
import sys

if __name__ == '__main__':
    CLASS = int(sys.argv[1])
    net = caffe.Net("./deploy_googlenetCAM.prototxt",
                    "./imagenet_googleletCAM_train_iter_120000.caffemodel",
                    caffe.TEST)
    net.forward()
    weights = net.params['CAM_fc'][0].data
    conv_img = net.blobs['CAM_conv'].data[0]
    weights = np.array(weights,dtype=np.float)
    conv_img = np.array(conv_img,dtype=np.float)
    heat_map = np.zeros([14,14],dtype = np.float)
    for i in range(1024):
        w = weights[CLASS][i]
        heat_map += w*conv_img[i]
    heat_map = cv2.resize(heat_map,(224,224))

    src = cv2.imread('./cat_dog.jpg')
    src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
    src = cv2.resize(src,(224,224))
    heat_map = 100*heat_map

    print net.blobs['prob'].data[0][CLASS]
    s = plt.imshow(src)
    s = plt.imshow(heat_map,alpha=0.5, interpolation='nearest')
    plt.show()


