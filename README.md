# SBD(Segmentation Based Detector)

SBD is an object detector based on segmentation.<br>

The output of SBD inner network is a one-channel image.<br>
In the output image, the part where trained objects are likely to exist is closer to white.<br>
And the part where there seems to be no trained object gets closer to black.<br>

SBD apply threshold to the output image and set it as the probability that you want to detect threshold value.<br>
Then, SBD detect contours from the binary image and return the bounding box values for the detected contours.<br>
To detect multiple objects in an image, SBD has multiple output layers.<br>
SBD, which detects one object, has one output layer and SBD, which detect three objects, has three output layers.<br>
Each output layer detects each class.<br>

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
The output of the network is a binary image as much as the number of trained classes.<br>
Before extracting the output, network use one filter and a 1 * 1 kernel to perform the convolution.<br>

<img src="/md/structure.jpg" width="800">
