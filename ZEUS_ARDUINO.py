import re
import serial
import time
import firebase_admin
from firebase_admin import credentials, db

# Firebase 초기화 (이미 초기화된 경우 재초기화 방지)
if not firebase_admin._apps:
    cred = credentials.Certificate(r"C:\Users\STORY\Desktop\필요한 파이썬 코드\zero-rbiz-45576-firebase-adminsdk-a5lp1-c1b58ea507.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://zero-rbiz-45576-default-rtdb.firebaseio.com/'
    })

# 시리얼 통신 설정
py_serial = serial.Serial(
    port='COM8',
    baudrate=9600,
    timeout=1
)
    
# Firebase에서 필요한 성분 비율 데이터를 가져오기
def get_required_ingredients():
    ref = db.reference('UserResults/requiredIngredients')
    ingredients_data = ref.get()
    return ingredients_data

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
    move_volume = abs(target_volume - current_volume)  # 이동할 용량 계산
    steps_per_50ul = 200  # 50 마이크로리터당 200 스텝
    steps = (move_volume // 50) * steps_per_50ul  # 필요한 스텝 수 계산
    return steps

# 피펫 용량 조절 함수
def adjust_pipette(target_volume, current_volume):
    # 시리얼 포트가 열리지 않았으면 모터 제어를 생략
    if not py_serial:
        print("시리얼 포트가 열리지 않아 모터를 제어할 수 없습니다.")
        return current_volume
    
    if target_volume == current_volume:
        print("현재 용량과 목표 용량이 동일합니다.")
        return current_volume

    steps = calculate_steps(target_volume, current_volume)
    direction = 'CW' if target_volume > current_volume else 'CCW'
    # 시리얼 통신으로 명령 전송
    command = f"{direction},{steps}\n"

    print(f"{direction} 방향으로 {steps} 스텝 이동 (각도: {steps * 1.8:.2f}도)")
    py_serial.write(command.encode())  # 명령 전송
    print(f"전송된 명령: {command}")
    
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

# 용액 비율에 맞춰 피펫 용량 조절
def distribute_liquid(volumes, current_volume):
    for liquid, volume in volumes.items():
        print(f"{liquid} 용액 {volume} 마이크로리터 담기 시작")

        while volume >= 1000:
            current_volume = adjust_pipette(1000, current_volume)
            print(f"{liquid} 용액 1000 마이크로리터 담음")
            volume -= 1000
            time.sleep(3)  # 대기 시간

        if volume > 0:
            current_volume = adjust_pipette(volume, current_volume)
            print(f"{liquid} 용액 {volume} 마이크로리터 담음")
            time.sleep(3)

        print(f"{liquid} 용액을 모두 옮김")
        print("-" * 50)

    current_volume = adjust_pipette(500, current_volume)
    print("피펫을 500 마이크로리터로 되돌림")
    return current_volume

# Firebase에서 데이터를 받아 모터 회전
ingredients_data = {'레티놀': '1400μl', '비타민': '1600μl'}
#ingredients_data = get_required_ingredients() #500->1000 ->400->1000->600->1000->500

if ingredients_data:
    ingredient_names = list(ingredients_data.keys())
    volumes = extract_numbers(ingredients_data)

    liquid_volumes = {name: vol for name, vol in zip(ingredient_names, volumes)}
    print(f"용액: {liquid_volumes}")

    distribute_liquid(liquid_volumes, 500)
else:
    print("Firebase에서 데이터를 가져오지 못했습니다.")