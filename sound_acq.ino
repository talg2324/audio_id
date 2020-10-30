
float mic_1;
float mic_1_per;
unsigned long int millisec;
unsigned long int micro;

void setup() {
  Serial.begin(128000);
  Serial.println("CLEARDATA");
  Serial.println("LABEL,CLOCK,micro,mic_1");
}

void loop() {
  mic_1 = analogRead(A0);
  micro = micros();

  Serial.print("DATA,TIME");
  Serial.print(",");
  Serial.print(micro);
  Serial.print(",");
  Serial.println(mic_1); 
   
}
