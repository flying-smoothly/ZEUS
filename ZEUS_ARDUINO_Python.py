import re
import serial
import time

def motor(vol,current_volume):

    # 시리얼 통신 설정
    py_serial = serial.Serial(port='/dev/rfcomm0', baudrate=9600, timeout=1)

    # 성분 비율에서 숫자 추출하기
    def extract_numbers(ingredients_data):
        try:
            ingredients_string = ' '.join(map(str, ingredients_data.values()))
            numbers = re.findall(r'(\d+)', ingredients_string)
            print(f"추출된 숫자: {numbers}")
            return [int(num) for num in numbers]
        except Exception as e:
            print(f"숫자 추출 중 오류 발생: {e}")
            return []

    # 스텝 수 계산 함수
    def calculate_steps(target_volume, current_volume):
        move_volume = abs(target_volume - current_volume)  
        steps_per_50ul = 200  # 50μl 당 200 스텝
        steps = (move_volume // 50) * steps_per_50ul 
        return steps

    # 피펫 용량 조절
    def adjust_pipette(target_volume, current_volume):
        if not py_serial or target_volume == current_volume:
            return current_volume

        steps = calculate_steps(target_volume, current_volume)
        direction = 'CW' if target_volume > current_volume else 'CCW'
        command = f"{direction},{steps}\n"

        py_serial.write(command.encode())  # 명령 전송    
        print(f"현재 용량: {current_volume}, 목표 용량: {target_volume}, 스텝 수: {steps}, 각도: {steps * 1.8:.2f}도")
        
        # 모터 응답 대기 (최대 5초)
        start_time = time.time()
        while time.time() - start_time <= 5:
            if py_serial.in_waiting > 0:
                response = py_serial.readline().decode().strip()
                print(f"모터 응답: {response}")
                break
        else:
            print("모터 응답 없음. 다음 단계로 넘어갑니다.")

        return target_volume

    # 피펫으로 용액 분배
    def distribute_liquid(vol, current_volume):
        while volume > 0:
            current_volume = adjust_pipette(volume, current_volume)
            print(f"{volume}μl 담음")
            time.sleep(10)

        current_volume = adjust_pipette(500, current_volume)
        print("피펫을 500μl로 되돌림")
        return current_volume

    # 실행 코드

distribute_liquid(vol_list, 500)

motor(vol,500)