#Arduinoからシリアル通信でデータを受信して、そのまま表示するプログラム
import serial
import matplotlib.pyplot as plt
#シリアル通信でデータを受信する
if __name__ == '__main__':
    ser = serial.Serial('/dev/tty.usbmodem1301', 9600)
    data = []
    while True:
        if ser.in_waiting > 0:
            row = ser.readline() # 1行読み取り
            text = row.rstrip().decode('utf-8') # 末尾の改行コードを除去
            print("data:", text)
            #リアルタイムでグラフを表示
            data.append(int(text))
            
            plt.ylim(0, 1024)
            plt.plot(data, color='red')
            plt.pause(0.01)
            if len(data) > 100:
                data.pop(0)
                plt.clf()
    
