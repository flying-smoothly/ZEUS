const int dirPin = 2;
const int stepPin = 3;
const int enablePin = 8;   // ENABLE 핀 (모터 드라이버 활성화/비활성화)

const int stepsPerRevolution = 200;  // 1회전(360도)에 필요한 스텝 수

void setup() {
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enablePin, OUTPUT);  // ENABLE 핀 출력 모드 설정

  // 모터 초기 상태 설정 (모터 정지)
  digitalWrite(enablePin, HIGH);  // 초기 상태: 모터 드라이버 비활성화
  digitalWrite(stepPin, LOW);
  digitalWrite(dirPin, LOW);

  Serial.begin(9600);  // 시리얼 통신을 9600 보드레이트로 설정
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');  // 개행 문자까지 읽음
    input.trim();  // 공백 제거

    int commaIndex = input.indexOf(',');  // 쉼표 위치 찾기
    if (commaIndex > 0) {
      String direction = input.substring(0, commaIndex);  // 방향 정보 (CW 또는 CCW)
      int steps = input.substring(commaIndex + 1).toInt();

      // 유효한 각도 값이면 모터 회전
      if (steps > 0) {
        digitalWrite(enablePin, LOW);  // 모터 드라이버 활성화
        rotateMotor(steps, direction);  // 입력된 각도와 방향에 따라 모터 회전
        digitalWrite(enablePin, HIGH);  // 모터 드라이버 비활성화
      } else {
        Serial.println("유효하지 않은 각도입니다.");  // 오류 메시지 출력
      }
    } else {
      Serial.println("유효하지 않은 입력 형식입니다.");  // 쉼표가 없을 경우 오류 메시지
    }
    
    Serial.flush();
    delay(500);
  }
}

// 모터 회전 함수
void rotateMotor(int steps, String direction) {
  digitalWrite(dirPin, direction == "CW" ? HIGH : LOW);

  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(1000);  // 펄스 유지 시간
    digitalWrite(stepPin, LOW);
    delayMicroseconds(1000);  // 펄스 간 간격
  }
  delay(500);  // 모터 회전 후 2초 대기
  Serial.println("모터 회전 완료");
}



