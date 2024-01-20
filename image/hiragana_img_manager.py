import cv2
import pandas as pd
class HiraganaImgManager:
    #ひらがなと画像の辞書
    hiragana_img_dict = {}
    df = pd.read_csv("../50on.csv",encoding="UTF-8", header=None) #実行するディレクトリによって変更

    def __init__(self):
        #ひらがなの画像を読み込んで辞書に格納
        for char in ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん","ー","小"]:
            self.hiragana_img_dict[char] = cv2.imread("../image/"+char+".png", cv2.IMREAD_UNCHANGED)#実行するディレクトリによって変更
    #画像の任意の位置にひらがなを貼り付ける
    def putHiragana(self, char, img, pos, size, alpha=1):
         # bgraに変換
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        # ひらがなの画像を取得
        hiragana_img = self.hiragana_img_dict[char]
        # ひらがな画像のサイズを変更
        hiragana_img = cv2.resize(hiragana_img, (size, size))
        # 貼り付け先領域の座標を計算
        x1, x2 = pos[0]-size/2, pos[0] + size/2
        y1, y2 = pos[1]-size/2, pos[1] + size/2
        #母音によって貼り付け先の座標を調整
        dan = None
        for index, row in self.df.iterrows():
            for col_label, cell_value in row.items():
                # セルの値と比較して一致する場合、列番号を取得
                if cell_value == char:
                    dan = col_label
                    break
        if dan == 1:
            #左
            x1 -= size
            x2 -= size
        elif dan == 2:
            #上
            y1 -= size
            y2 -= size
        elif dan == 3:
            #右
            x1 += size
            x2 += size
        elif dan == 4:
            #下
            y1 += size
            y2 += size
        #imgの解像度を取得
        height, width = img.shape[:2]
        #imgの範囲外の場合は調整する
        if y1 < 0:
            y1 = 0
            y2 = size
        if y2 > height:
            y1 = height - size
            y2 = height
        if x1 < 0:
            x1 = 0
            x2 = size
        if x2 > width:
            x1 = width - size
            x2 = width
        # 貼り付け先領域のアルファチャンネルを計算
        alpha_s = alpha * hiragana_img[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        x1 = int(x1)  
        x2 = int(x2)  
        y1 = int(y1)  
        y2 = int(y2)  
        # 画像を重ねる
        for c in range(0, 3):
            img[y1:y2, x1:x2, c] = (alpha_s * hiragana_img[:, :, c] + alpha_l * img[y1:y2, x1:x2, c])
            
        # bgrに戻す
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img






        

    
        
