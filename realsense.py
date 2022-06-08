import pyrealsense2 as rs
import cv2, math
from decimal import Decimal
from decimal import *
import numpy as np
import os
import pycuda.autoinit
import time
from PIL import Image

def main() :
    theta = 0
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)
        
    # get camera intrinsics
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    
    #=========== Yolov4 TensorRt ağırlıkları yüklenmektedir =======================
    
    weightsPath_tiny = "yolov4-tiny-taco.weights"
    configPath_tiny = "yolov4-tiny-taco.cfg"

    tiny_net = cv2.dnn.readNet(weightsPath_tiny, configPath_tiny)
    tiny_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    tiny_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    model = cv2.dnn_DetectionModel(tiny_net)
    
 
    
    def YOLOv4_video(pred_image):
        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
        image_test = cv2.cvtColor(pred_image, cv2.COLOR_RGBA2RGB)
        image = image_test.copy()
        print('image',image.shape)
        confThreshold= 0.5
        nmsThreshold = 0.4
        classes, confidences, boxes = model.detect(image, confThreshold, nmsThreshold)
        
        return classes,confidences,boxes
        
        
    key = ' '
    LABELS = [  'Aluminium foil',
                'Battery',
                'Aluminium blister pack',
                'Carded blister pack',
                'Other plastic bottle',
                'Clear plastic bottle',
                'Glass bottle',
                'Plastic bottle cap',
                'Metal bottle cap',
                'Broken glass',
                'Food Can',
                'Aerosol',
                'Drink can',
                'Toilet tube',
                'Other carton',
                'Egg carton',
                'Drink carton',
                'Corrugated carton',
                'Meal carton',
                'Pizza box',
                'Paper cup',
                'Disposable plastic cup',
                'Foam cup',
                'Glass cup',
                'Other plastic cup',
                'Food waste',
                'Glass jar',
                'Plastic lid',
                'Metal lid',
                'Other plastic',
                'Magazine paper',
                'Tissues',
                'Wrapping paper',
                'Normal paper',
                'Paper bag',
                'Plastified paper bag',
                'Plastic film',
                'Six pack rings',
                'Garbage bag',
                'Other plastic wrapper',
                'Single-use carrier bag',
                'Polypropylene bag',
                'Crisp packet',
                'Spread tub',
                'Tupperware',
                'Disposable food container',
                'Foam food container',
                'Other plastic container',
                'Plastic glooves',
                'Plastic utensils',
                'Pop tab',
                'Rope & strings',
                'Scrap metal',
                'Shoe',
                'Squeezable tube',
                'Plastic straw',
                'Paper straw',
                'Styrofoam piece',
                'Unlabeled litter',
                'Cigarette']

    COLORS = [[0, 0, 255]]
    prev_frame_time=0
    new_frame_time=0
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        #depth_image = np.asanyarray(depth_frame.get_data())
            
        image = Image.fromarray(color_image)
        img = np.asarray(image)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        classes,confidences,boxes = YOLOv4_video(img)
        print("predict:",classes,boxes)
        if len(boxes)>0:
            for cl,score,(x_min,y_min,x_max,y_max) in zip(classes,confidences,boxes):
                start_pooint = (int(x_min),int(y_min))
                end_point = (int(x_max),int(y_max))
                
                x = int(x_min +( x_max-x_min)/2)
                y = int(y_min + (y_max-y_min)/2)
                color = COLORS[0]
                img =cv2.rectangle(img,start_pooint,end_point,color,3)
                img = cv2.circle(img,(x,y),5,[0,0,255],5)
                text = f'{LABELS[int(cl)]}: {score:0.2f}'
                cv2.putText(img,text,(int(x_min),int(y_min-7)),cv2.FONT_ITALIC,1,COLORS[0],2 )
                
                x = round(x)
                y = round(y)
                dist = depth_frame.get_distance(int(x), int(y))*1000 #convert to mm

                #calculate real world coordinates
                Xtemp = dist*(x -intr.ppx)/intr.fx
                Ytemp = dist*(y -intr.ppy)/intr.fy
                Ztemp = dist

                Xtarget = Xtemp - 35 #35 is RGB camera module offset from the center of the realsense
                Ytarget = -(Ztemp*math.sin(theta) + Ytemp*math.cos(theta))
                Ztarget = Ztemp*math.cos(theta) + Ytemp*math.sin(theta)

                coordinates_text = "(" + str(Decimal(str(Xtarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                                    ", " + str(Decimal(str(Ytarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + \
                                    ", " + str(Decimal(str(Ztarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)) + ")"

                coordinat = (Decimal(str(Ztarget)).quantize(Decimal('0'), rounding=ROUND_HALF_UP))
                print("Distance to Camera at (class : {0}, score : {1:0.2f}): distance : {2:0.2f} mm".format(LABELS[int(cl)], score, coordinat), end="\r")
                cv2.putText(img,"Distance: "+str(round(coordinat,2))+'m',(int(x_max-180),int(y_max+30)),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
                
                new_frame_time=time.time()
                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                
                print('FPS : %.2f  ' % fps)
                cv2.imshow("Image", img)
                    
        else:
            cv2.imshow("Image", img)
            #cv2.imshow("Depth", depth_image_ocv)
            
        cv2.waitKey(1)


    cv2.destroyAllWindows()

    print("\nFINISH")

if __name__ == "__main__":
    main()