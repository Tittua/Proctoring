#importing libraries
import mediapipe as mp
import cv2
import time
import numpy as np

#Opening the camera
cam=cv2.VideoCapture(0)

mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5)

#Setting up drawing utils
mp_drawing=mp.solutions.drawing_utils

drawing_spec=mp_drawing.DrawingSpec(thickness=1,circle_radius=1)




while cam.isOpened():
    sucess,image=cam.read()
    
    start=time.time()

    #Flipping the image for lateral inversion
    image=cv2.flip(image,1)
    
    #Color inversion
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #To create  the face mesh
    result=face_mesh.process(image)

    #setting parameter that the image can only be read
    image.flags.writeable=False

    #Storing image dimensions
    img_w,img_h,img_c=image.shape
    face_3d=[]
    face_2d=[]


    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            for idx,lm in enumerate(face_landmarks.landmarks):
                if idx==33 or idx==263 or idx==1  or idx==61 or idx==291 or idx==199:
                    if idx==1:
                        nose_2d=(lm.x*img_w,lm.y*img_h)
                        nose_3d=(lm.x*img_w,lm.y*img_h,lm.z*3000)
                        x,y=int(lm.x*img_w),int(lm.y*img_h)

                        #Get  the 2D cordinates
                        face_2d.append([x,y])

                        #3D dimension
                        face_3d.append([x,y,lm.z])

                #Converting into numpy array
                face_2d=np.array(face_2d,dtype=np.float64)

                face_3d=np.array(face_3d,dtype=np.float64)

                #camera matrix
                focal_length=1*img_w
                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])
                
                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            #to find the angle of movement of the face we need to find the rotation vector
            rmat,jac=cv2.Rodrigues(rot_vec)

            #angle extraction
            angles,mtxR,mtxQ,Qx,Qy,Qz=cv2.RQDecomp3x3(rmat)

            #Get the rotations angles
            x=angles[0]*360
            y=angles[1]*360
            z=angles[2]*360

            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -10:
                text = "Looking Down"
            elif x > 10:
                text = "Looking Up"
            else:
                text = "Forward"
            






