using OpenCvSharp;
using OpenCvSharp.Dnn;
using System;
using System.Collections.Generic;

namespace SbdTester
{
    class Sbd
    {
        private readonly Net net;
        private readonly Size inputSize;
        private readonly int outputWidth;
        private readonly int outputHeight;
        private readonly int classCount;
        private readonly int boxPaddingVal = 1;

        private readonly double fontScale = 0.4;
        private readonly Scalar black = new Scalar(0, 0, 0);
        private readonly Scalar white = new Scalar(255, 255, 255);

        private readonly string[] classNames;

        private readonly Scalar[] boxColors = {
            new Scalar(255, 0, 0),
            new Scalar(255, 255, 0),
            new Scalar(0, 255, 0),
            new Scalar(0, 255, 255),
            new Scalar(255, 0, 255),
            new Scalar(255, 255, 0),
            new Scalar(212, 255, 127),
            new Scalar(196, 228, 255),
            new Scalar(226, 43, 138),
            new Scalar(42, 42, 165),
            new Scalar(135, 184, 222),
            new Scalar(160, 158, 95),
            new Scalar(0, 255, 127),
            new Scalar(30, 105, 210),
            new Scalar(80, 127, 255),
            new Scalar(237, 149, 100),
            new Scalar(220, 248, 255),
            new Scalar(60, 20, 220),
            new Scalar(255, 255, 0),
            new Scalar(139, 0, 0),
            new Scalar(139, 139, 0),
            new Scalar(11, 134, 184),
            new Scalar(169, 169, 169),
            new Scalar(0, 100, 0),
            new Scalar(169, 169, 169),
            new Scalar(107, 183, 189),
            new Scalar(139, 0, 139),
            new Scalar(47, 107, 85),
            new Scalar(0, 140, 255),
            new Scalar(204, 50, 153),
            new Scalar(0, 0, 139),
            new Scalar(122, 150, 233),
            new Scalar(143, 188, 143),
            new Scalar(139, 61, 72),
            new Scalar(79, 79, 47),
            new Scalar(79, 79, 47),
            new Scalar(209, 206, 0),
            new Scalar(211, 0, 148),
            new Scalar(147, 20, 255),
            new Scalar(255, 191, 0),
            new Scalar(105, 105, 105),
            new Scalar(105, 105, 105),
            new Scalar(255, 144, 30),
            new Scalar(34, 34, 178),
            new Scalar(240, 250, 255),
            new Scalar(34, 139, 34),
            new Scalar(255, 0, 255),
            new Scalar(220, 220, 220),
            new Scalar(255, 248, 248),
            new Scalar(0, 215, 255),
            new Scalar(32, 165, 218),
            new Scalar(128, 128, 128),
            new Scalar(0, 128, 0),
            new Scalar(47, 255, 173),
            new Scalar(128, 128, 128),
            new Scalar(240, 255, 240),
            new Scalar(180, 105, 255),
            new Scalar(92, 92, 205),
            new Scalar(130, 0, 75),
            new Scalar(240, 255, 255),
            new Scalar(140, 230, 240),
            new Scalar(250, 230, 230),
            new Scalar(245, 240, 255),
            new Scalar(0, 252, 124),
            new Scalar(205, 250, 255),
            new Scalar(230, 216, 173),
            new Scalar(128, 128, 240),
            new Scalar(255, 255, 224),
            new Scalar(210, 250, 250),
            new Scalar(211, 211, 211),
            new Scalar(144, 238, 144),
            new Scalar(211, 211, 211),
            new Scalar(193, 182, 255),
            new Scalar(122, 160, 255),
            new Scalar(170, 178, 32),
            new Scalar(250, 206, 135),
            new Scalar(153, 136, 119),
            new Scalar(153, 136, 119),
            new Scalar(222, 196, 176),
            new Scalar(224, 255, 255),
            new Scalar(0, 255, 0),
            new Scalar(50, 205, 50),
            new Scalar(230, 240, 250),
            new Scalar(255, 0, 255),
            new Scalar(0, 0, 128),
            new Scalar(170, 205, 102),
            new Scalar(205, 0, 0),
            new Scalar(211, 85, 186),
            new Scalar(219, 112, 147),
            new Scalar(113, 179, 60),
            new Scalar(238, 104, 123),
            new Scalar(154, 250, 0),
            new Scalar(204, 209, 72),
            new Scalar(133, 21, 199),
            new Scalar(112, 25, 25),
            new Scalar(250, 255, 245),
            new Scalar(225, 228, 255),
            new Scalar(181, 228, 255),
            new Scalar(173, 222, 255),
            new Scalar(128, 0, 0),
            new Scalar(230, 245, 253),
            new Scalar(0, 128, 128),
            new Scalar(35, 142, 107),
            new Scalar(0, 165, 255),
            new Scalar(0, 69, 255),
            new Scalar(214, 112, 218),
            new Scalar(170, 232, 238),
            new Scalar(152, 251, 152),
            new Scalar(238, 238, 175),
            new Scalar(147, 112, 219),
            new Scalar(213, 239, 255),
            new Scalar(185, 218, 255),
            new Scalar(63, 133, 205),
            new Scalar(203, 192, 255),
            new Scalar(221, 160, 221),
            new Scalar(230, 224, 176),
            new Scalar(128, 0, 128),
            new Scalar(0, 0, 255),
            new Scalar(143, 143, 188),
            new Scalar(225, 105, 65),
            new Scalar(19, 69, 139),
            new Scalar(114, 128, 250),
            new Scalar(96, 164, 244),
            new Scalar(87, 139, 46),
            new Scalar(238, 245, 255),
            new Scalar(45, 82, 160),
            new Scalar(192, 192, 192),
            new Scalar(235, 206, 135),
            new Scalar(205, 90, 106),
            new Scalar(144, 128, 112),
            new Scalar(144, 128, 112),
            new Scalar(250, 250, 255),
            new Scalar(127, 255, 0),
            new Scalar(180, 130, 70),
            new Scalar(140, 180, 210),
            new Scalar(128, 128, 0),
            new Scalar(216, 191, 216),
            new Scalar(71, 99, 255),
            new Scalar(208, 224, 64),
            new Scalar(238, 130, 238),
            new Scalar(179, 222, 245),
            new Scalar(255, 255, 255),
            new Scalar(245, 245, 245),
            new Scalar(0, 255, 255),
            new Scalar(50, 205, 154)
        };

        public class BoundingBox
        {
            public int classIndex;
            public int x;
            public int y;
            public int w;
            public int h;

            public BoundingBox(int classIndex, int x1, int y1, int x2, int y2)
            {
                this.classIndex = classIndex;
                x = Math.Min(x1, x2);
                y = Math.Min(y1, y2);
                w = Math.Abs(x2 - x1);
                h = Math.Abs(y2 - y1);
            }

            public override string ToString()
            {
                return $"[classIndex : {classIndex}, box: [{x}, {y}, {w}, {h}]]";
            }
        }

        public Sbd(string modelPath, Size inputSize, int outputWidth, int outputHeight, string[] classNames)
        {
            net = CvDnn.ReadNet(modelPath);
            if (net.Empty())
            {
                Console.WriteLine("fail to load model");
                return;
            }
            net.SetPreferableBackend(Net.Backend.OPENCV);
            net.SetPreferableTarget(Net.Target.CPU);
            this.inputSize = inputSize;
            this.outputWidth = outputWidth;
            this.outputHeight = outputHeight;
            this.classNames = classNames;
            this.classCount = classNames.Length;
        }

        public List<BoundingBox> Inference(Mat x)
        {
            if (x.Empty())
            {
                Console.WriteLine("empty x");
                return null;
            }
            Size rawSize = x.Size();
            x = CvDnn.BlobFromImage(x, scaleFactor: 1 / 255.0f, size: inputSize, swapRB: false, crop: false);
            net.SetInput(x);
            Mat res = net.Forward();
            if (res.Empty())
            {
                Console.WriteLine("empty result");
                return null;
            }
            List<BoundingBox> boxes = new List<BoundingBox>();
            for (int i = 0; i < classCount; ++i)
            {
                Mat classResult = new Mat(outputHeight, outputWidth, MatType.CV_8UC1);
                for (int j = 0; j < outputHeight; ++j)
                {
                    for (int k = 0; k < outputWidth; ++k)
                    {
                        float val = res.At<float>(0, i, j, k);
                        classResult.At<uint>(j, k) = val > 0.5 ? (uint)255 : 0;
                    }
                }
                Cv2.Resize(classResult, classResult, rawSize);
                Point[][] contours = Cv2.FindContoursAsArray(classResult, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
                foreach (Point[] contour in contours)
                {
                    Rect rect = Cv2.BoundingRect(contour);
                    int x1 = rect.TopLeft.X - boxPaddingVal;
                    int y1 = rect.TopLeft.Y - boxPaddingVal;
                    int x2 = rect.BottomRight.X + boxPaddingVal;
                    int y2 = rect.BottomRight.Y + boxPaddingVal;
                    boxes.Add(new BoundingBox(i, x1, y1, x2, y2));
                }
            }
            boxes.Sort((BoundingBox a, BoundingBox b) => { return a.x - b.x; });
            return boxes;
        }

        public Mat DrawBoxes(Mat raw, List<BoundingBox> boxes)
        {
            foreach (BoundingBox box in boxes)
            {
                string className = classNames[box.classIndex];
                Scalar boxColor = boxColors[box.classIndex];
                Scalar fontColor = IsBright(boxColor) ? black : white;
                int[] labelWidthHeight = CalculateLabelWidthHeight(className);
                Cv2.Rectangle(raw, new Rect(box.x, box.y, box.w, box.h), boxColor, thickness: 2, lineType: LineTypes.AntiAlias);
                Cv2.Rectangle(raw, new Rect(box.x - 1, box.y - labelWidthHeight[1], labelWidthHeight[0], labelWidthHeight[1]), boxColor, thickness: -1, lineType: LineTypes.AntiAlias);
                Cv2.PutText(raw, className, new Point(box.x + 2, box.y - 5), HersheyFonts.HersheyDuplex, fontScale, fontColor, thickness: 1, lineType: LineTypes.AntiAlias);
            }
            return raw;
        }

        private bool IsBright(Scalar bgr)
        {
            Mat tmp = new Mat(1, 1, MatType.CV_8UC3);
            Cv2.Rectangle(tmp, new Rect(0, 0, 1, 1), bgr, thickness: -1);
            Cv2.CvtColor(tmp, tmp, ColorConversionCodes.BGR2GRAY);
            return tmp.At<Vec3b>(0).Item0 > 127;
        }

        private int[] CalculateLabelWidthHeight(string className)
        {
            Mat zeros = Mat.Zeros(new Size(500, 50), MatType.CV_8UC1);
            Cv2.PutText(zeros, className, new Point(50, 50), HersheyFonts.HersheyDuplex, fontScale, white, thickness: 1, lineType: LineTypes.AntiAlias);
            Cv2.Resize(zeros, zeros, new Size(0, 0), fx: 0.5, fy: 0.5);
            Cv2.Dilate(zeros, zeros, Mat.Ones(new Size(2, 3), MatType.CV_8UC1));
            Cv2.Resize(zeros, zeros, new Size(0, 0), fx: 2, fy: 2);
            Cv2.Threshold(zeros, zeros, 1, 255, ThresholdTypes.Binary);
            Point[][] contours = Cv2.FindContoursAsArray(zeros, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
            Rect rect = Cv2.BoundingRect(contours[0]);
            return new int[] { rect.Width + 2, rect.Height + 2 };
        }
    }
}
