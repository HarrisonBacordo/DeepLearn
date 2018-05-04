from DeepLearn.cnn import ConvLayer, PoolingLayer
import numpy as np

c1 = ConvLayer.ConvLayer(["test.png"], 3, [5, 5])
filtered_imgs = c1.execute()
p1 = PoolingLayer.PoolLayer(filtered_imgs, [2, 2], 2)
pooled_imgs = p1.execute()
c2 = ConvLayer.ConvLayer(pooled_imgs, 3, [5, 5])
filtered_imgs = c2.execute()
p2 = PoolingLayer.PoolLayer(filtered_imgs, [2, 2], 2)
pooled_imgs = p2.execute()
pooled_imgs.

print()
