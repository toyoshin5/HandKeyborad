#日本語を表示する関数
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

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

def putText_japanese_multi(img, texts, points, size, color):
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
    for text, point in zip(texts, points):
        draw.text(point, text, fill=color, font=font)
    #PILからndarrayに変換して返す
    return np.array(img_pil)

#imageに日本語を描画
def draw_shiin(image,res,hand_randmark):
    #辞書を作成
    shiin_center_dict = {}
    shiin_center_dict["あ"] = [((hand_randmark[8].x+hand_randmark[7].x)/2),((hand_randmark[8].y+hand_randmark[7].y)/2)]
    shiin_center_dict["か"] = [((hand_randmark[7].x+hand_randmark[6].x)/2),((hand_randmark[7].y+hand_randmark[6].y)/2)]
    shiin_center_dict["さ"] = [((hand_randmark[6].x+hand_randmark[5].x)/2),((hand_randmark[6].y+hand_randmark[5].y)/2)]
    shiin_center_dict["た"] = [((hand_randmark[12].x+hand_randmark[11].x)/2),((hand_randmark[12].y+hand_randmark[11].y)/2)]
    shiin_center_dict["な"] = [((hand_randmark[11].x+hand_randmark[10].x)/2),((hand_randmark[11].y+hand_randmark[10].y)/2)]
    shiin_center_dict["は"] = [((hand_randmark[10].x+hand_randmark[9].x)/2),((hand_randmark[10].y+hand_randmark[9].y)/2)]
    shiin_center_dict["ま"] = [((hand_randmark[16].x+hand_randmark[15].x)/2),((hand_randmark[16].y+hand_randmark[15].y)/2)]
    shiin_center_dict["や"] = [((hand_randmark[15].x+hand_randmark[14].x)/2),((hand_randmark[15].y+hand_randmark[14].y)/2)]
    shiin_center_dict["ら"] = [((hand_randmark[14].x+hand_randmark[13].x)/2),((hand_randmark[14].y+hand_randmark[13].y)/2)]
    shiin_center_dict["小"] = [((hand_randmark[20].x+hand_randmark[19].x)/2),((hand_randmark[20].y+hand_randmark[19].y)/2)]
    shiin_center_dict["わ"] = [((hand_randmark[19].x+hand_randmark[18].x)/2),((hand_randmark[19].y+hand_randmark[18].y)/2)]
    fontsize = 30
    #解像度に合わせて座標を変換
    for key in shiin_center_dict:
        shiin_center_dict[key] = (int(shiin_center_dict[key][0]*res[0]-fontsize/2),int(shiin_center_dict[key][1]*res[1]-fontsize/2))
    #描画
    image = putText_japanese_multi(image, list(shiin_center_dict.keys()), list(shiin_center_dict.values()), fontsize, (255,255,255))
    return image

 #[0,0]以上res以下に収まるように調整する関数
def adjust_point(point,res):
    if point[0]<0:
        point[0] = 0
    elif point[0]>res[0]:
        point[0] = res[0]
    if point[1]<0:
        point[1] = 0
    elif point[1]>res[1]:
        point[1] = res[1]
    return point
def draw_hiragana(image,res,hand_randmark,isAll,shiin,chr,udlr):
    #辞書を作成
    shiin_center_dict = {}
    shiin_center_dict["あ"] = [((hand_randmark[8].x+hand_randmark[7].x)/2),((hand_randmark[8].y+hand_randmark[7].y)/2)]
    shiin_center_dict["か"] = [((hand_randmark[7].x+hand_randmark[6].x)/2),((hand_randmark[7].y+hand_randmark[6].y)/2)]
    shiin_center_dict["さ"] = [((hand_randmark[6].x+hand_randmark[5].x)/2),((hand_randmark[6].y+hand_randmark[5].y)/2)]
    shiin_center_dict["た"] = [((hand_randmark[12].x+hand_randmark[11].x)/2),((hand_randmark[12].y+hand_randmark[11].y)/2)]
    shiin_center_dict["な"] = [((hand_randmark[11].x+hand_randmark[10].x)/2),((hand_randmark[11].y+hand_randmark[10].y)/2)]
    shiin_center_dict["は"] = [((hand_randmark[10].x+hand_randmark[9].x)/2),((hand_randmark[10].y+hand_randmark[9].y)/2)]
    shiin_center_dict["ま"] = [((hand_randmark[16].x+hand_randmark[15].x)/2),((hand_randmark[16].y+hand_randmark[15].y)/2)]
    shiin_center_dict["や"] = [((hand_randmark[15].x+hand_randmark[14].x)/2),((hand_randmark[15].y+hand_randmark[14].y)/2)]
    shiin_center_dict["ら"] = [((hand_randmark[14].x+hand_randmark[13].x)/2),((hand_randmark[14].y+hand_randmark[13].y)/2)]
    shiin_center_dict["だ"] = [((hand_randmark[20].x+hand_randmark[19].x)/2),((hand_randmark[20].y+hand_randmark[19].y)/2)]
    shiin_center_dict["わ"] = [((hand_randmark[19].x+hand_randmark[18].x)/2),((hand_randmark[19].y+hand_randmark[18].y)/2)]
    fontsize = 50
    #解像度に合わせて座標を変換
    for key in shiin_center_dict:
        shiin_center_dict[key] = (int(shiin_center_dict[key][0]*res[0]-fontsize/2),int(shiin_center_dict[key][1]*res[1]-fontsize/2))
    if(not isAll):
        #todo:リリース時に指定されたboinを描画する
        #shiinのudlrの方向にchrを描画
        if(udlr == 2):
            pos = [shiin_center_dict[shiin][0],shiin_center_dict[shiin][1]-fontsize*2]
            pos = adjust_point(pos,res)
            image = putText_japanese(image, chr, pos, fontsize, (255,255,255))
        elif(udlr == 4):
            pos = [shiin_center_dict[shiin][0],shiin_center_dict[shiin][1]+fontsize*2]
            pos = adjust_point(pos,res)
            image = putText_japanese(image, chr, pos, fontsize, (255,255,255))
        elif(udlr == 3):
            pos = [shiin_center_dict[shiin][0]+fontsize*2,shiin_center_dict[shiin][1]]
            pos = adjust_point(pos,res)
            image = putText_japanese(image, chr, pos, fontsize, (255,255,255))
        elif(udlr == 1):
            pos = [shiin_center_dict[shiin][0]-fontsize*2,shiin_center_dict[shiin][1]]
            pos = adjust_point(pos,res)
            image = putText_japanese(image, chr, pos, fontsize, (255,255,255))
        elif(udlr == 0):
            pos = [shiin_center_dict[shiin][0],shiin_center_dict[shiin][1]]
            pos = adjust_point(pos,res)
            image = putText_japanese(image, chr, pos, fontsize, (255,255,255))
    return image
        
        