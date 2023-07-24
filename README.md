# cleverest_test

This repository contains the code for a drone detection application using computer vision techniques. 
The application processes a video, detects small objects in it, and highlights them using bounding boxes.

The application has the following requirements:

Python (>=3.6)
OpenCV (>=4.0)

Features:


1.Background subtraction to distinguish foreground objects from the background.
2.Thresholding to convert the foreground mask into a binary image.
3.Morphological operations to remove noise and merge nearby regions.
4.Contour detection to identify potential small objects.
5.Bounding box visualization to highlight the detected small objects in the video.
