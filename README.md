# SBD(Segmentation Based Detector)

SBD is an object detector based on segmentation.<br>
The image is divided into a grid of n * m, such as yolo, and each grid contains the probability that an object exists.<br>

What's different from yolo is that SBD labels only the probability that objects will be included.<br>
The output of SBD is a one-channel image of n * m.<br>
In the output image, the part where trained objects are likely to exist is closer to white.<br>
And the part where there seems to be no trained object gets closer to black.<br>

You can apply threshold to the output image and set it as the probability that you want to detect threshold value.<br>

Then, the binary image detects the contour and returns the bound box value for the detected contour.<br>
<br>

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
Pass the results from predict to the sbd.bounding_box function.
```python
>>> img = cv2.imread('img.jpg')
>>> result = sbd.predict(img)
>>> bounding_box_imgage = sbd.bounding_box(img, result)
>>> cv2.imshow('img', bounding_box_image)
```
Then you can get a labeled image.<br>
<img src="/md/res.jpg" width="500">

## Structure
The output value is a one-channel image of n * m.<br>
Before extracting the output, use one filter and a 1 * 1 kernel to perform the convolution.<br>

The number of output layers is the same as the number of training classes and generates training data for each class in the sbd.load function when loading training data.<br>
<img src="/md/structure.jpg" width="800">
