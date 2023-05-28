// 50msごとにFSR402の値をシリアルモニタ出力するプログラム
const int vol_pin = 1;

int vol_value = 0;

void setup() {
  Serial.begin( 9600 );
}

void loop() {
  vol_value = analogRead( vol_pin );
  Serial.println(vol_value);
  delay( 50 );
}
