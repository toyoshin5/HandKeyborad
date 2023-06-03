
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
def putText_japanese(img, text, point, size, color):
    #hiragino font
    try:
        fontpath = '/System/Library/Fonts/ヒラギノ角ゴシック W4.ttc'
        font = ImageFont.truetype(fontpath, size)
    except:
        print("フォントが見つかりませんでした。: " + fontpath)
        sys.exit()
    #imgをndarrayからPILに変換
    img_pil = Image.fromarray(img)
    #drawインスタンス生成
    draw = ImageDraw.Draw(img_pil)
    #テキスト描画
    draw.text(point, text, fill=color, font=font)
    #PILからndarrayに変換して返す
    return np.array(img_pil)

if __name__ == "__main__":
    #../50on.csvを読み込み
    df = pd.read_csv("../50on.csv",encoding="UTF-8", header=None)

    #あ~んまでfor
    for char in ["あ","い","う","え","お","か","き","く","け","こ","さ","し","す","せ","そ","た","ち","つ","て","と","な","に","ぬ","ね","の","は","ひ","ふ","へ","ほ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん","だ"]:
        dan = None
        for index, row in df.iterrows():
            for col_label, cell_value in row.iteritems():
                # セルの値と比較して一致する場合、列番号を取得
                if cell_value == char:
                    dan = col_label
                    break
        #400*400の透明な画像を生成
        canvas = np.zeros((400, 400, 4), dtype=np.uint8)
        canvas[:, :, 3] = 0  # アルファチャンネルを完全に透明に設定
        if dan == 0:
            pts = np.array([[300,100],[300,300],[100,300],[100,100]], np.int32)
        elif dan == 1:
            pts = np.array([[100,100],[300,100],[400,200],[300,300],[100,300]], np.int32)
        elif dan == 2:
            pts = np.array([[100,100],[300,100],[300,300],[200,400],[100,300]], np.int32)
        elif dan == 3:
            pts = np.array([[100,100],[300,100],[300,300],[100,300],[0,200]], np.int32)
        elif dan == 4:
            pts = np.array([[100,100],[200,0],[300,100],[300,300],[100,300]], np.int32)
        pts = pts.reshape((-1,1,2))
        #白で塗りつぶし
        cv2.fillPoly(canvas, [pts], color=(255,255,255,255))
        #ひらがなを描画
        fontsize = 100
        if char=="だ":
            char = "小"
        canvas = putText_japanese(canvas, char, (150, 150), fontsize, (0,0,0,255))
        #画像を保存
        cv2.imwrite(char+".png", canvas)
    
