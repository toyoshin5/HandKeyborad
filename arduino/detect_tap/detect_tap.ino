// タップを検出してシリアルに出力するプログラム
const int vol_pin = 1;
int vol_value = 0;
bool isOn = false;

//フリックの方向を正確に識別するために
//指を離してから信号を送信するまでに遅延を導入
void setup() {
  Serial.begin(9600);
}

void loop() {
  vol_value = analogRead( vol_pin );
  if(vol_value < 1000){
    if(!isOn){
      Serial.println("tap");
      isOn = true;
    }
  }
  if(vol_value >= 1023){
    if(isOn){
      delay(50);
      Serial.println("release");
      isOn = false;
    }
  }
  delay( 50 );
}
