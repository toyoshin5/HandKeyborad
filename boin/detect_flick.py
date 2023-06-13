import mediapipe as mp
import numpy as np
import cv2
import serial

ARDUINO_PATH = "/dev/cu.usbmodem1301" #Arduinoのシリアルポート

def draw_vector(image,vec,res,origin=[0.5,0.5]):
    vec = [vec[0]*res[0],vec[1]*res[1]]
    origin = [origin[0]*res[0],origin[1]*res[1]]
    image = cv2.arrowedLine(image, (int(origin[0]),int(origin[1])), (int(origin[0]+vec[0]),int(origin[1]+vec[1])), (0, 255, 0), thickness=2)
    return image


if __name__ == '__main__':
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.9,
    )
    mp_drawing = mp.solutions.drawing_utils
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(0)
    resolution = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    #シリアル通信の設定
    ser = serial.Serial(ARDUINO_PATH, 9600)
    #押下時の親指のx,y座標を格納
    thumb_tap = np.array([0,0])
    #リリース時の親指のx,y座標を格納
    thumb_release = np.array([0,0])
    #移動量のx,y座標を格納
    thumb_move = np.array([0,0])
    #タップしているか(Debug用)
    is_tapping = False
    origin = np.array([0,0])
    #タップしたときに親指から最も近い点の番号を格納
    min_idx = 0

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            #親指を青くする
            image = cv2.circle(image,(int(hand_landmarks.landmark[4].x*resolution[0]),int(hand_landmarks.landmark[4].y*resolution[1])), 10, (255,0,0), -1)
            #最も近い点を赤くする
            image = cv2.circle(image,(int(hand_landmarks.landmark[min_idx].x*resolution[0]),int(hand_landmarks.landmark[min_idx].y*resolution[1])), 10, (0,0,255), -1)
            
            if ser.in_waiting > 0:
                row = ser.readline()
                msg = row.decode('utf-8').rstrip()
                if msg == "tap":
                    is_tapping = True
                    thumb_tap_org = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
                    #親指から最も近い点を探す
                    min_dist = 100
                    for i in range(5,21):
                        dist = np.linalg.norm(np.array([hand_landmarks.landmark[i].x,hand_landmarks.landmark[i].y])-np.array(thumb_tap_org))
                        if dist < min_dist:
                            min_dist = dist
                            min_idx = i
                    origin = [hand_landmarks.landmark[min_idx].x,hand_landmarks.landmark[min_idx].y]
                    thumb_tap = np.array(thumb_tap_org)-np.array(origin)
                    
                if msg == "release":  
                    is_tapping = False
                    thumb_release_org = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
                    origin = [hand_landmarks.landmark[min_idx].x,hand_landmarks.landmark[min_idx].y]
                    thumb_release = np.array(thumb_release_org)-np.array(origin)
                    thumb_move = np.array(thumb_release)-np.array(thumb_tap)
                    thumb_move_res = np.array([thumb_move[0]*resolution[0],thumb_move[1]*resolution[1]])
                    #親指の動きが小さい場合
                    if abs(thumb_move_res[0]) < 25 and abs(thumb_move_res[1]) < 25:
                        print("タップ")
                    else:
                        #上下左右を判定
                        vec = np.array([1,0])
                        cos_theta = np.dot(vec,thumb_move)/(np.linalg.norm(vec)*np.linalg.norm(thumb_move))
                        gaiseki = np.cross(vec,thumb_move,)
                        if(gaiseki > 0):
                            theta = np.arccos(cos_theta)*180/np.pi
                        else:
                            theta = 360-np.arccos(cos_theta)*180/np.pi
                        if theta < 45 or theta > 315:
                            print("右")
                        elif theta < 135:
                            print("下")
                        elif theta < 225:
                            print("左")
                        else:
                            print("上") 
                    print(str(thumb_move[0]*resolution[0])+","+str(thumb_move[1]*resolution[1]))
            draw_vector(image,thumb_move,resolution,origin)
        cv2.imshow('MediaPipe Hands', image)
        kry = cv2.waitKey(1)
    hands.close()
    cap.release()
