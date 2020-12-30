# SBD(Segmentation Based Detector)

SBD is an object detector based on segmentation.<br>

The output of SBD inner network is a n-classes channel images.<br>
In the output image, the part where trained objects are likely to exist is closer to white.<br>
And the part where there seems to be no trained object gets closer to black.<br>

SBD apply threshold to the output image and set it as the probability that you want to detect threshold value.<br>
Then, SBD detect contours from the binary image and return the bounding box values for the detected contours.<br>
To detect multiple objects in an image, SBD has multiple output channels.<br>
SBD, which detects one object, has one output channel and SBD, which detect three objects, has three output channel.<br>
Each channel detects each class.<br>

Shit. I don't know what you're talking about to say.<br>
Let me explain more.<br>

SBD output gives the probability map that objects will be included in the corresponding grid cell, not in the 0 and 1 binaries.<br>
Therefore, the closer the label is to the actual object segmentation, the more accurate the model is.<br>

This is a completely different mechanism from YOLO or SSD, a popular object detection algorithm.<br>
SBD does not have channels of x, y, with, height and objectness.<br>

So how do you extract the location of the object?<br>
The key to SBD is to use the probability map converted to channel-first-ordering as an image.<br>

Depending on default keras channel ordering, the output is returned in channel-last-ordering format.<br>
Converts output to channel-first-ordering form through conversion methods such as np.moveaxis or np.transpose.<br>

After converting to an image by multiplying 255, use the contour detection method in opencv to determine the size of the object.<br>
Resize the results to the original image size for more accurate results.<br>

This is not just about resizing images.<br>
It is meaningful to increase the size of the probabilities printed through the bilinear interpolation and to interpolate the values between them.<br>

In case the object is caught too large or too small, SBD provides class percentage threshold.<br>
After binary processing based on threshold of interpolated output, detect the contour.<br>
Get more sophisticated bounding boxes.<br>

The detected contour x, y, width, and height are objects from the actual input image.<br>

If libraries such as opencv are not available, the same contour x, y, with, and height can be obtained through DFS algorithms, etc.<br>

In environments where performance and memory are critical, such as embedded, we recommend using DFS to get bounding boxes.<br>
If this is not the case, get a bounding box through bilinear interpolation.<br>

## Loss function
Basically, you can use the binary crossentropy loss function to train.<br>
It is suitable for having sigmoid output of 0 and 1.<br>

<img src="/md/bce_formula.jpg" width="500"><br>

Bce is a loss function for binary classification.<br>
Thus, assuming that there are two classes, the loss is strong enough to converge to zero or one.<br>

<img src="/md/bce_graph.jpg" width="200"><br>

Below is the bce loss value for label 1.<br>

```python
>>> y_true, y_pred = [[1.0]], [[0.0]]
>>> print(tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy())
>>> 15.4249
>>>
>>> y_true, y_pred = [[1.0]], [[0.5]]
>>> print(tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy())
>>> 0.6931
>>>
>>> y_true, y_pred = [[1.0]], [[0.9]]
>>> print(tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy())
>>> 0.1054
```

But there is one big problem with bce.<br>
If you label consecutive numbers, such as 0.03, 0.45, and 0.81, rather than 0 and 1 binaries.<br>

```python
>>> y_true, y_pred = [[0.5]], [[0.0]]
>>> print(tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy())
>>> 7.7175
>>>
>>> y_true, y_pred = [[0.5]], [[0.5]]
>>> print(tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy())
>>> 0.6931
>>>
>>> y_true, y_pred = [[0.5]], [[0.9]]
>>> print(tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy())
>>> 1.2040
```

It does not appear to be suitable for SBD, which directly labels the probability that objects exist in each grid.<br>
For example, if 0.5 is labeled, bce will cause strong losses to continue converging to zero or one,<br>
which can severely interfere with the way it descends.<br>

Therefore, you may not be able to learn as you want.<br>

We customized the loss function for proper training.<br>
This is a method of log loss of an absolute error value of the actual label and output value,<br>
rather than the actual output value of the loss function.<br>

<img src="/md/male_formula.jpg" width="500"><br>

This can take the gradient and loss values of the bce loss function<br>
and at the same time give a stable loss in consecutive numbers between 0 and 1.

<img src="/md/male_graph.jpg" width="200"><br>

```python
>>> y_true, y_pred = [[1.0]], [[0.0]]
>>> print(MeanAbsoluteLogError()(y_true, y_pred).numpy())
>>> 15.9424
>>>
>>> y_true, y_pred = [[1.0]], [[0.5]]
>>> print(MeanAbsoluteLogError()(y_true, y_pred).numpy())
>>> 0.6931
>>>
>>> y_true, y_pred = [[1.0]], [[0.9]]
>>> print(MeanAbsoluteLogError()(y_true, y_pred).numpy())
>>> 0.1054
>>>
>>> y_true, y_pred = [[0.5]], [[0.0]]
>>> print(MeanAbsoluteLogError()(y_true, y_pred).numpy())
>>> 0.6931
>>>
>>> y_true, y_pred = [[0.5]], [[0.5]]
>>> print(MeanAbsoluteLogError()(y_true, y_pred).numpy())
>>> 0.0000
>>>
>>> y_true, y_pred = [[0.5]], [[0.9]]
>>> print(MeanAbsoluteLogError()(y_true, y_pred).numpy())
>>> 0.5108
```

## Labeling

SBD is fully compatible with the yolo label.<br>
Refer to [**LabelImg**](https://github.com/tzutalin/labelImg) for image labeling.<br>
<br>
<img src="/md/labelimg.jpg" width="800">

## Result
The result value includes the bounding box coordinates(x1, y1, x2, y2) and class of the detected object.
```python
>>> result = sbd.predict(cv2.imread('img.jpg'))
>>> print(result)
[
    {
        "class": 2,
        "box": [
            127,
            215,
            316,
            548
        ]
    },
    {
        "class": 0,
        "box": [
            131,
            134,
            571,
            424
        ]
    },
    {
        "class": 1,
        "box": [
            464,
            68,
            692,
            175
        ]
    }
]
```

## Bounding box
Pass the results from forward to the sbd.bounding_box function.
```python
>>> img = cv2.imread('img.jpg')
>>> result = sbd.forward(img)
>>> bounding_box_imgage = sbd.bounding_box(img, result)
>>> cv2.imshow('img', bounding_box_image)
```
Then you can get a labeled image.<br>
<img src="/md/res.jpg" width="500">

## Structure
The output of the network is a binary image as much as the number of trained classes.<br>
Before extracting the output, network use one filter and a 1 * 1 kernel to perform the convolution.<br>

<img src="/md/structure.jpg" width="800">
