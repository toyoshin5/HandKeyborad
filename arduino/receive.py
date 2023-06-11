#Arduinoからシリアル通信でデータを受信して、そのまま表示するプログラム
import serial
import matplotlib.pyplot as plt
#シリアル通信でデータを受信する
if __name__ == '__main__':
    ser = serial.Serial('/dev/tty.usbmodem1301', 9600)
    data = []
    th = 900
    while True:
        if ser.in_waiting > 0:
            row = ser.readline() # 1行読み取り
            text = row.rstrip().decode('utf-8') # 末尾の改行コードを除去
            print("data:", text)
            #dataがint型に変換できないなら無視
            try:     
                data.append(int(text))
            except ValueError:
                continue
            plt.ylim(0, 1023)
            color = "green" if int(text) <= th else "red"
            plt.plot(data, color=color)
            plt.pause(0.01)
            if len(data) > 100:
                data.pop(0)
                plt.clf()
    