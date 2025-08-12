import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../')
from utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    def __init__(self,frame, method: str = 'homography'):
        self.minimum_distance = 5
        self.method = method  # 'homography' or 'lk'

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,0:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )

        # ORB for homography-based global motion
        self._orb = cv2.ORB_create(nfeatures=1000)

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def _estimate_motion_lk(self, frames):
        camera_movement = [[0,0]]*len(frames)
        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)

            old_gray = frame_gray.copy()
        return camera_movement

    def _estimate_motion_homography(self, frames):
        camera_movement = [[0,0]]*len(frames)
        prev_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        for frame_num in range(1,len(frames)):
            gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            kp1, des1 = self._orb.detectAndCompute(prev_gray, None)
            kp2, des2 = self._orb.detectAndCompute(gray, None)
            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                prev_gray = gray
                continue
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            if len(matches) < 10:
                prev_gray = gray
                continue
            matches = sorted(matches, key=lambda m: m.distance)[:200]
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is None:
                prev_gray = gray
                continue
            tx, ty = float(H[0,2]), float(H[1,2])
            camera_movement[frame_num] = [tx, ty]
            prev_gray = gray
        return camera_movement

    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        if self.method == 'homography':
            camera_movement = self._estimate_motion_homography(frames)
            # Fallback if homography failed to produce any movement
            if not any(cm != [0,0] for cm in camera_movement):
                camera_movement = self._estimate_motion_lk(frames)
        else:
            camera_movement = self._estimate_motion_lk(frames)

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self,frames, camera_movement_per_frame):
        output_frames=[]

        for frame_num, frame in enumerate(frames):
            frame= frame.copy()

            overlay = frame.copy()
            cv2.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha =0.6
            cv2.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv2.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame) 

        return output_frames