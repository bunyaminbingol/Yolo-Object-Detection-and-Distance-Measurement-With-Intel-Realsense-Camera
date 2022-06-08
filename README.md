# Yolo-Object-Detection-and-Distance-Measurement-With-Intel-Realsense-Camera
You can use this repo to detect object with Intel realsense camera and measure the distance of this object.


### How to use 

You need to run this script like that `python zed.py `
or if you use tensorRT yolo, You need to run this script like that `python realsense_trt.py `
You need to edit the codes in `realsense.py` line according to yourself.

specify the yolo weights and config files you trained before.
~~~~~~~~~~~~
29. weightsPath = "yolov4-tiny-taco.weights"
30. configPath = "yolov4-tiny-taco.cfg"
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

then you need to edit the following lines in `realsense.py` 

Size should be changed according to your config file
~~~~~~
40. model.setInputParams(size=(608, 608), scale=1/255, swapRB=True)
~~~~~~~~~~~~~~~~~~~~
Edit them according to your class labels.
~~~~~~~~~~~~
52.  LABELS = [ 'class_name_1',
                'class_name_2',
                'class_name_3',
                'class_name_4',
                .
                .
                'class_name_n'
                ]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
that's all, if you have a intel realsense camera you can easily find the distance of the objects you have detected
## You can see how the program works in the gif below.

![into gif](https://github.com/bunyaminbingol/Yolo-Object-Detection-and-Distance-Measurement-With-Intel-Realsense-Camera/blob/main/intro.gif)

I used the https://github.com/MehmetOKUYAR/Yolo-Object-Detection-and-Distance-Measurement-with-Zed-camera repo while creating this work
