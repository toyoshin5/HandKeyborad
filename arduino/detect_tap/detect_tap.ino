// タップを検出してシリアルに出力するプログラム
const int vol_pin = 1;

int vol_value = 0;
int vol_value_old = 0;
void setup() {
  Serial.begin( 9600 );
}

void loop() {
  vol_value = analogRead( vol_pin );
  if(vol_value < 1023){
    if(vol_value_old >= 1023){
      Serial.println("tap");
    }
  }
  vol_value_old = vol_value;
  delay( 50 );
}
