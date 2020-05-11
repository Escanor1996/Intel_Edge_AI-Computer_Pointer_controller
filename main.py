import cv2
import os
import logging
import numpy as np
from face_detection import Model_Face_Detection
from facial_landmarks_detection import Model_Facial_Landmarks_Detection
from gaze_estimation import Model_Gaze_Estimation
from head_pose_estimation import Model_Head_Pose_Estimation
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--facedetectionmodel", required=True, type=str,
                        help=" Path to .xml file of Face Detection model.")
    parser.add_argument("-fl", "--faciallandmarkmodel", required=True, type=str,
                        help=" Path to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--headposemodel", required=True, type=str,
                        help=" Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gazeestimationmodel", required=True, type=str,
                        help=" Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help=" Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--previewFlags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Seperated by space)"
                             "for see the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="path of extensions if any layers is incompatible with  hardware")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold for model to identify the face .")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to run on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "(CPU by default)")
    
    return parser



def main():

    args = build_argparser().parse_args()
    previewFlags = args.previewFlags
    
    logger = logging.getLogger()
    inputFile = args.input
    inputFeeder = None

    if inputFile.lower()=="cam":
        inputFeeder=InputFeeder("cam")
    if not os.path.isfile(inputFile):
        logger.error("Unable to find input file")
        exit(1)
    inputFeeder=InputFeeder("video",inputFile)


    mfd=Model_Face_Detection(args.facedetectionmodel,args.device,args.cpu_extension)
    mfld=Model_Facial_Landmarks_Detection(args.faciallandmarkmodel,args.device,args.cpu_extension)
    mge=Model_Gaze_Estimation(args.gazeestimationmodel,args.device,args.cpu_extension)
    mhpe=Model_Head_Pose_Estimation(args.headposemodel,args.device,args.cpu_extension)

    mc = MouseController('medium','fast')
    #inputFeeder=InputFeeder("cam")
    inputFeeder.load_data()

    mfd.load_model()
    mfld.load_model()
    mge.load_model()
    mhpe.load_model()
    frame_count = 0
    for ret, frame in inputFeeder.next_batch():


        if frame is not None:

            frame_count+=1
            if frame_count%5==0:
                cv2.imshow('video',cv2.resize(frame,(500,500)))
        
            key = cv2.waitKey(60)
            croppedFace, face_coords = mfd.predict(frame.copy(), args.prob_threshold)
            if type(croppedFace)==int:
                logger.error("No face detected.")
                if key==27:
                    break
                continue
            
            hp_out = mhpe.predict(croppedFace.copy())
            
            left_eye, right_eye, eye_coords = mfld.predict(croppedFace.copy())
            #print(left_eye
            
            new_mouse_coord, gaze_vector = mge.predict(left_eye, right_eye, hp_out)
            
            if (not len(previewFlags)==0):
                preview_frame = frame.copy()
                if 'fd' in previewFlags:
                
                    preview_frame = croppedFace
                if 'fld' in previewFlags:
                    cv2.rectangle(croppedFace, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                    cv2.rectangle(croppedFace, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)
                    
                    
                if 'hp' in previewFlags:
                    cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0],hp_out[1],hp_out[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
                if 'ge' in previewFlags:
                    x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                    le =cv2.line(left_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                    cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                    re = cv2.line(right_eye.copy(), (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                    cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                    croppedFace[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                    croppedFace[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                    
                    
                cv2.imshow("visualization",cv2.resize(preview_frame,(500,500)))
            
            if frame_count%5==0:
                mc.move(new_mouse_coord[0],new_mouse_coord[1])    
            if key==27:
                    break
    logger.error("video ended...")
    cv2.destroyAllWindows()
    inputFeeder.close()
     
    

if __name__ == '__main__':
    main()


