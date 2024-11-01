import re
import serial
import time 

# 시리얼 통신 설정py_serial = serial.Serial(port='/dev/rfcomm0', baudrate=9600, timeout=1)
py_serial = serial.Serial(port='COM5', baudrate=9600, timeout=1)
py_serial.reset_input_buffer()

def calculate_steps(target_volume, current_volume):
    move_volume = abs(target_volume - current_volume)  
    steps_per_50ul = 200  # 50μl 당 200 스텝
    steps = (move_volume // 50) * steps_per_50ul
    return steps

def adjust_pipette(target_volume, current_volume):
    if target_volume == current_volume:
        return current_volume

    steps = calculate_steps(target_volume, current_volume)
    direction = 'CW' if target_volume > current_volume else 'CCW'
    command = f"{direction},{steps}\n"

    retries = 1
    for _ in range(retries):
        py_serial.write(command.encode())
        time.sleep(0.5)  # 응답 대기   
    print(f"현재 용량: {current_volume}, 목표 용량: {target_volume}, "
          f"스텝 수: {steps}, 각도: {steps * 1.8:.2f}도")

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

def motor(vol, current_volume):
    # 한 번에 1000μl 주입
    current_volume = adjust_pipette(vol, current_volume)
    print(f"{vol}μl 담음")
    time.sleep(1)  # 10초 대기

    # 피펫을 500μl로 되돌림
    current_volume = adjust_pipette(500, current_volume)
    print("피펫을 500μl로 되돌림")
    return current_volume

# 실행 코드
vol=1000
motor(vol, 500)