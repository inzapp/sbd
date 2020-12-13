# SBD(Segmentation Based Detector)

SBD is an object detector based on segmentation.<br>

The output of SBD inner network is a n-classes channel images.<br>
In the output image, the part where trained objects are likely to exist is closer to white.<br>
And the part where there seems to be no trained object gets closer to black.<br>

SBD apply threshold to the output image and set it as the probability that you want to detect threshold value.<br>
Then, SBD detect contours from the binary image and return the bounding box values for the detected contours.<br>
To detect multiple objects in an image, SBD has multiple output layers.<br>
SBD, which detects one object, has one channel output and SBD, which detect three objects, has three channel output.<br>
Each channel detects each class.<br>

Shit. I don't know what you're talking about to say.<br>
Let me explain more.<br>

In case you detect objects that are both small and small in the grid size.<br>
SBD output gives the probability that objects will be included in the corresponding grid cell, not in the 0 and 1 binaries.<br>
Therefore, the closer the label is to the actual object segmentation, the more accurate the model is.<br>

The output of SBD is the probability map of that class, which consists of values between 0 and 1.<br>

This can look exactly the same as class probabilities in YOLO.<br>
However, SBD does not have channels of x, y, with, height and objectness.<br>

So how do you extract the location of the object?<br>
The key to SBD is to use the probability map converted to channel-first-ordering as an image.<br>

Depending on default keas channel ordering, the output is returned in channel-last-ordering format.<br>
Converts output to channel-first-ordering form through conversion methods such as np.moveaxis or np.transpose.<br>

After converting to an image by multiplying 255, use the contour detection method in opencv to determine the size of the object.<br>
Resize the results to the original image size for more accurate results.<br>

This is not just about resizing images.<br>
It is meaningful to increase the size of the probabilities printed through the billainar interpolation technique and to interpolate the values between them.<br>

In case the object is caught too large or too small, SBD provides class percentage threshold.<br>
After binary processing based on threshold of interpolated output, detect the contour.<br>
Get more sophisticated bounding boxes.<br>

The detected contour x, y, width, and height are objects from the actual input image.<br>

If libraries such as opencv are not available, the same contour x, y, with, and height can be obtained through DFS algorithms, etc.<br>

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
