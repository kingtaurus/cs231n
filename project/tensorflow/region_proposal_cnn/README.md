Neural Network Project to Detect License Plates
===============================================

Material
--------
[Number Plate Recognition](http://matthewearl.github.io/2016/05/06/cnn-anpr/)
[deep-anpr](https://github.com/matthewearl/deep-anpr)

Details
-------
Start with the initial material as a guide for trying different object location methods in neural networks. The initial approach uses sliding window plus a deep neural network to detect images.

Papers and Code
---------------
[OverFeat](http://arxiv.org/abs/1312.6229)
[OverFeat Code #1](https://github.com/sermanet/OverFeat)
[OverFeat Code #2](https://github.com/jhjin/overfeat-torch)
[YOLO](http://pjreddie.com/media/files/papers/yolo.pdf)
[Faster-RCNN](http://arxiv.org/abs/1506.01497)
[FASTER-RCNN CODE](https://github.com/rbgirshick/py-faster-rcnn)
[Fast-RCNN](http://arxiv.org/pdf/1504.08083v2)
[FAST-RNN CODE](https://github.com/rbgirshick/fast-rcnn)
[RCNN](https://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr.pdf)
[RCNN CODE](https://github.com/rbgirshick/rcnn)

Fonts
-----
[Font Classifications](https://www.fonts.com/content/learning/fontology/level-1/type-anatomy/type-classifications)
`/usr/share/wine/fonts/*.ttf`
`usr/share/fonts/truetype/freefont/*.ttf`


Work
----
I've reimplemented a few of the features. I'm just going to try using the default wallpapers from the Ubuntu apt repository (instead of using SUN image database).

Image Collection:
=================

[SUN Image Set](http://groups.csail.mit.edu/vision/SUN/)
[UKNumberPlate.tff](http://www.dafont.com/uk-number-plate.font)

```
curl -O -J -L http://dl.dafont.com/dl/?f=uk_number_plate
```
