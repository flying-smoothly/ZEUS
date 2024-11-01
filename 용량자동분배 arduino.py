import re
import serial
import time

# 시리얼 통신 설정 (포트와 보드레이트 설정)
py_serial = serial.Serial(
    port='COM5',  # 블루투스 시리얼 포트로 연결하기
    baudrate=9600,
    timeout=1  # 읽기 시 타임아웃 설정 (1초)
)

# 딕셔너리로 직접 성분 데이터 입력
ingredients_data = {
    '레티놀': 4,
    '비타민C': 3,
    '티트리': 2,
    '히알루론산': 1
}

# 성분 비율에서 숫자 추출하기
def extract_numbers(ingredients_data):
    # 딕셔너리 값들을 문자열로 합침
    ingredients_string = ' '.join(map(str, ingredients_data.values()))
    # 문자열에서 숫자 추출
    numbers = re.findall(r'(\d+)', ingredients_string)
    print(f"추출된 숫자: {numbers}")  # 추출된 숫자 확인
    return [int(num) for num in numbers]  # 문자열 숫자를 정수로 변환하여 반환

# 피펫 용량 조절 함수
def adjust_pipette(target_volume, current_volume):
    angle_per_ul = 7.2  # 1마이크로리터당 회전 각도 ###########################

    if target_volume < current_volume:
        direction = 'CW'
        angle = (current_volume - target_volume) * angle_per_ul + 36
        print("**시계방향 회전**")
    elif target_volume > current_volume:
        direction = 'CCW'
        angle = (target_volume - current_volume) * angle_per_ul + 36
        print("**반시계방향 회전**")
    else:
        return current_volume

    print(f"현재 용량: {current_volume}, 목표 용량: {target_volume}")

    # 아두이노에 각도 전송
    py_serial.write(f"{direction},{angle}\n".encode())
    time.sleep(2)
    return target_volume

# 용액 비율에 맞춰 피펫 용량 조절
def distribute_liquid(volumes, current_volume):
    for liquid, volume in volumes.items():
        print(f"{liquid} 용액 {volume} 마이크로리터 담기 시작")

        if volume >= 1000:
            current_volume = adjust_pipette(1000, current_volume)
            print(f"{liquid} 용액 1000 마이크로리터 담음")
            count = int(volume // 1000)
            for _ in range(count - 1):
                print("반복")
                time.sleep(10)
            volume -= 1000 * count

        if volume > 0:
            current_volume = adjust_pipette(volume, current_volume)
            print(f"{liquid} 용액 {volume} 마이크로리터 담음")
            time.sleep(10)

        current_volume = adjust_pipette(0, current_volume)
        print(f"{liquid} 용액 모두 옮김")
        print("-" * 50)
        time.sleep(10)

    current_volume = adjust_pipette(500, current_volume)
    print("피펫을 500 마이크로리터로 되돌림")
    return current_volume

# 메인 코드 실행
if ingredients_data:
    ratios = extract_numbers(ingredients_data)
    
    total_volume = 4000  # 총 용량
    sum_ratios = sum(ratios) if sum(ratios) > 0 else 1  # 나눗셈 오류 방지

    liquid_volumes = {}
    for i, ratio in enumerate(ratios):
        liquid_name = f"Liquid-{i + 1}"
        volume = total_volume * ratio / sum_ratios
        liquid_volumes[liquid_name] = volume
        
        print(f"용액: {liquid_volumes}")

        distribute_liquid(liquid_volumes, 500)

