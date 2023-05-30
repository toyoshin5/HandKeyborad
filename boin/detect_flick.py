import mediapipe as mp
import numpy as np
import cv2
import serial

ARDUINO_PATH = "/dev/cu.usbmodem101" #Arduinoのシリアルポート

def draw_vector(image,vec,res,origin=[0.5,0.5]):
    vec = [vec[0]*res[0],vec[1]*res[1]]
    origin = [origin[0]*res[0],origin[1]*res[1]]
    image = cv2.arrowedLine(image, (int(origin[0]),int(origin[1])), (int(origin[0]+vec[0]),int(origin[1]+vec[1])), (0, 255, 0), thickness=2)
    return image


if __name__ == '__main__':
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    #ウェブカメラからの入力
    cap = cv2.VideoCapture(0)
    resolution = [cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)]
    #シリアル通信の設定
    ser = serial.Serial(ARDUINO_PATH, 9600)
    #直近5フレームの親指のx,y変位を格納するリスト
    thumb_history = [[0,0],[0,0],[0,0],[0,0],[0,0]]
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
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
            #8,12,16,10の平均を原点とする
            origin = [(hand_landmarks.landmark[8].x+hand_landmarks.landmark[12].x+hand_landmarks.landmark[16].x+hand_landmarks.landmark[10].x)/4,(hand_landmarks.landmark[8].y+hand_landmarks.landmark[12].y+hand_landmarks.landmark[16].y+hand_landmarks.landmark[10].y)/4]
            thumb  = [hand_landmarks.landmark[4].x,hand_landmarks.landmark[4].y]
            #原点からの親指のx,y座標と変位を計算
            thumb = [thumb[0]-origin[0],thumb[1]-origin[1]]
            #直近5フレームの親指のx,y座標,変位を格納するリストに左から追加
            thumb_history.insert(0,thumb)
            thumb_history.pop()
            vec = [hand_landmarks.landmark[12].x-hand_landmarks.landmark[9].x,hand_landmarks.landmark[12].y-hand_landmarks.landmark[9].y]
            image = draw_vector(image,vec,resolution,origin)

        cv2.imshow('MediaPipe Hands', image)
        kry = cv2.waitKey(1)
        #Arduinoからのシリアル通信を受け取る
        if results.multi_hand_landmarks:
            if ser.in_waiting > 0:
                row = ser.readline()
                msg = row.decode('utf-8').rstrip()
                if msg == "release":  
                    thumb_move = [thumb_history[0][0]-thumb_history[2][0],thumb_history[0][1]-thumb_history[2][1]] #直近3フレームの親指のx,y変位
                    #上下左右を判定
                    cos_theta = np.dot(vec,thumb_move)/(np.linalg.norm(vec)*np.linalg.norm(thumb_move))
                    gaiseki = np.cross(vec,thumb_move)
                    if(gaiseki > 0):
                        theta = np.arccos(cos_theta)*180/np.pi
                    else:
                        theta = 360-np.arccos(cos_theta)*180/np.pi
                    if theta < 45 or theta > 300:
                        print("左")
                    elif theta < 135:
                        print("下")
                    elif theta < 225:
                        print("右")
                    else:
                        print("上") 
    hands.close()
    cap.release()
