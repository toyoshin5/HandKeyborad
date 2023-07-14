// タップを検出してシリアルに出力するプログラム
#include <CapacitiveSensor.h>
bool isOn = false;
CapacitiveSensor   cs_4_2 = CapacitiveSensor(4,2);
//フリックの方向を正確に識別するために
//指を離してから信号を送信するまでに遅延を導入
void setup() {
  cs_4_2.set_CS_AutocaL_Millis(0xFFFFFFFF);     // turn off autocalibrate on channel 1 - just as an example
  Serial.begin(9600);
}

void loop() {
  long total = cs_4_2.capacitiveSensor(30);      // check on performance in milliseconds                   // tab character for debug windown spacing                  // print sensor output 1                            // arbitrary delay to limit data to serial port
  long vol_value = total;
  if (vol_value > 1000) {
      if (!isOn) {
          Serial.println("tap");
          isOn = true;
      }
  }
  if(vol_value < 500){
    if(isOn){
      delay(100);
      Serial.println("release");
      isOn = false;
    }
  }
  delay( 50 );
}
