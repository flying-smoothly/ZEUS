import cv2
import numpy as np
import pyrealsense2 as rs
import json
import uuid
import time
import requests
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

api_url = 'https://7x3edq0zuo.apigw.ntruss.com/custom/v1/35237/5667ef73dac709effcfdc518a98ea9238365ec62b0eb5f478392e28818a2efd5/general'
secret_key = 'eUx4Ulp3TnNOS1daSG9veVpGZEtybmRoalpMZlZQeFY='

# RealSense 파이프라인 시작
pipeline = rs.pipeline()
config = rs.config()

# 색상과 깊이 스트림 활성화 
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30) 
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# 파이프라인 시작
profile = pipeline.start(config)

# Spatial 필터 설정 (edge-aware) test.py로부터 필터 수치 변경 및 temporal 필터 추가.
spatial = rs.spatial_filter()        # Spatial Edge-Preserving filter
spatial.set_option(rs.option.holes_fill, 2)        # 오차를 줄이고 원래 깊이 값 보존
spatial.set_option(rs.option.filter_smooth_alpha, 0.6)  # 정적 환경용
spatial.set_option(rs.option.filter_smooth_delta, 15)   # 민감한 깊이 측정

# Temporal 필터 설정 (프레임 간 안정성 강화)
temporal = rs.temporal_filter()
temporal.set_option(rs.option.filter_smooth_alpha, 0.85)  # 정적 환경에 적합한 높은 Alpha 값
temporal.set_option(rs.option.filter_smooth_delta, 20)  # 작은 깊이 변화에도 민감하게 반응

# depth-to-disparity 변환 필터 추가
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

# Pillow로 이미지에 한글 텍스트 그리기 함수
def draw_text_with_pillow(image, text, position, font_path="NanumGothic.ttf", font_size=20):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=(0, 255, 0))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# Rigid body transformation 함수 정의
def apply_rigid_body_transform(point, rotation_matrix, translation_vector):
    point = np.array(point)
    transformed_point = np.dot(rotation_matrix, point) + translation_vector
    return transformed_point

last_ocr_time = time.time()

recognized_texts = set()

try:
    while True:
        # 프레임 세트 수집
        frameset = pipeline.wait_for_frames()

        # 깊이 프레임을 기준으로 정렬
        align = rs.align(rs.stream.color)
        frameset = align.process(frameset)

        # 정렬된 컬러 및 깊이 프레임 가져오기
        aligned_depth_frame = frameset.get_depth_frame()  # 정렬된 깊이 프레임
        aligned_color_frame = frameset.get_color_frame()  # 정렬된 컬러 프레임

        # Depth를 Disparity로 변환
        disparity_frame = depth_to_disparity.process(aligned_depth_frame)

        # Spatial Filter 적용 (edge-aware)
        filtered_depth = spatial.process(disparity_frame)

        # Temporal 필터 적용 (프레임 간 노이즈 감소)
        filtered_depth = temporal.process(filtered_depth)

        # 다시 Depth로 변환
        depth_frame_filtered = disparity_to_depth.process(filtered_depth)

        # 필터가 적용된 깊이 프레임 데이터를 NumPy 배열로 변환
        final_depth_frame = np.asanyarray(depth_frame_filtered.get_data())

        # 정렬된 컬러 프레임을 NumPy 배열로 변환
        aligned_color_image = np.asanyarray(aligned_color_frame.get_data())

        # 5초에 한 번씩 OCR 요청 수행
        if time.time() - last_ocr_time >= 3 and len(recognized_texts) < 4:
            # OCR 요청 데이터 생성
            _, encoded_img = cv2.imencode('.jpg', aligned_color_image)  # 이미지를 JPEG로 인코딩
            encoded_img = np.array(encoded_img).tobytes()  # 바이트 형태로 변환

            files = [('file', ('image.jpg', encoded_img, 'image/jpeg'))]
            request_json = {
                'images': [
                    {
                        'format': 'jpg',
                        'name': 'demo'
                    }
                ],
                'requestId': str(uuid.uuid4()),
                'version': 'V2',
                'timestamp': int(round(time.time() * 1000))
            }

            payload = {'message': json.dumps(request_json).encode('UTF-8')}
            headers = {'X-OCR-SECRET': secret_key}

            response = requests.post(api_url, headers=headers, data=payload, files=files)
            ocr_result = response.json()
            last_ocr_time = time.time()
            target_texts = ['티트리', '히알', '비타민', '레티놀']

            # OCR 결과에서 텍스트 및 경계 상자 추출
            pixels = []
            texts=[]

            for i in ocr_result['images'][0]['fields']:
                text = i['inferText']

                # 한글 텍스트 인코딩 확인
                try:
                    decoded_text = text.encode('utf-8').decode('utf-8')
                    # print(f"인식된 텍스트: {decoded_text}")
                except UnicodeDecodeError as e:
                    print(f"인코딩 오류 발생: {e}")

                # 각 target_text와 비교하여 일치하는 경우만 처리
                for target_text in target_texts:
                    if target_text.lower().strip() in text.lower().strip():
                        recognized_texts.add(target_text)  # 대소문자 구분 없이 비교, 앞뒤 공백 제거
                        bounding_box = i['boundingPoly']['vertices']

                        # 경계 상자의 중심 좌표 계산
                        center_x = np.mean([vertex['x'] for vertex in bounding_box])
                        center_y = np.mean([vertex['y'] for vertex in bounding_box])

                        # 중심 좌표에 해당하는 깊이 값 추출
                        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
                        if 0 <= int(center_y) < final_depth_frame.shape[0] and 0 <= int(center_x) < final_depth_frame.shape[1]:
                            center_depth_value = final_depth_frame[int(center_y), int(center_x)].astype(float) * depth_scale

                            # 3D 포인트 생성 (x, y, z) 좌표
                            pixel = [center_x, center_y, center_depth_value, text]
                            if text in target_texts:  # target_texts 리스트에 있는 경우만 저장
                                pixels.append(pixel)
                                texts.append(text)

                            print(f"감지된 텍스트: '{text}'")

                            # 경계 상자 그리기
                            pts = np.array([[vertex['x'], vertex['y']] for vertex in bounding_box], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(aligned_color_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                            # Pillow를 사용해 텍스트 표시
                            x, y = pts[0][0]
                            aligned_color_image = draw_text_with_pillow(aligned_color_image, text, (x, y - 10))

        # 이미지 시각화 (실시간)
        aligned_color_image_resized = cv2.resize(aligned_color_image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow('RealSense', aligned_color_image_resized)

        # 네 개의 target_texts가 모두 인식되면 종료
        if len(recognized_texts) == 4:
            print("모든 타겟 텍스트를 인식했습니다. 프로세스를 종료합니다.")
            break

        # q 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # RealSense 파이프라인 정지
    pipeline.stop()
    cv2.destroyAllWindows()

color_intrinsics = aligned_color_frame.profile.as_video_stream_profile().intrinsics
intrin = {
    'model': 'RS2_DISTORTION_INVERSE_BROWN_CONRADY',
    'ppx': color_intrinsics.ppx,
    'ppy': color_intrinsics.ppy,
    'fx': color_intrinsics.fx,
    'fy': color_intrinsics.fy,
    'coeffs': color_intrinsics.coeffs
}

# Deprojection 함수
def rs2_deproject_pixel_to_point(intrin, pixels):
    points = np.zeros((len(pixels), 3))  # NumPy 배열로 초기화
    
    # 모든 왜곡 계수가 0인지 확인
    distortion_is_zero = np.all(np.array(intrin['coeffs']) == 0)

    for i, pixel in enumerate(pixels):
        u, v, depth = pixel[:3]  # 픽셀 좌표와 깊이 값 분리

        if depth == 0 or np.isnan(depth):  # 유효하지 않은 깊이 값 처리
            continue

        # 픽셀 좌표를 정규화
        x = (u - intrin['ppx']) / intrin['fx']
        y = (v - intrin['ppy']) / intrin['fy']

        # 왜곡 보정이 필요한 경우 처리
        if not distortion_is_zero and intrin['model'] == 'RS2_DISTORTION_INVERSE_BROWN_CONRADY':
            r2 = x * x + y * y
            f = 1 + intrin['coeffs'][0] * r2 + intrin['coeffs'][1] * r2 * r2 + intrin['coeffs'][4] * r2 * r2 * r2
            ux = x * f + 2 * intrin['coeffs'][2] * x * y + intrin['coeffs'][3] * (r2 + 2 * x * x)
            uy = y * f + 2 * intrin['coeffs'][3] * x * y + intrin['coeffs'][2] * (r2 + 2 * y * y)
            x, y = ux, uy

        # 3D 포인트 계산
        points[i] = [depth * x, depth * y, depth]

    return points

points = rs2_deproject_pixel_to_point(intrin, pixels)

# Define rigid body transformation function
def apply_rigid_body_transform(point, rotation_matrix, translation_vector):
    # Convert point to numpy array if not already
    point = np.array(point)

    # Apply rigid body transformation: R * point + T
    transformed_point = np.dot(rotation_matrix, point) + translation_vector
    return transformed_point

# Example extrinsics from depth to color stream
rotation_matrix = [
    0.999944269657135, -0.00910048745572567, -0.005356862209737301,
    0.009115878492593765, 0.999954342842102, 0.002855878323316574,
    0.005330628249794245, -0.0029045515693724155, 0.9999815821647644
]

# Reshape rotation matrix to 3x3 matrix
R = np.array(rotation_matrix).reshape(3, 3)

translation_vector = [-0.014955335296690464, -3.531932452460751e-05, -0.00015003549924585968]
T = np.array(translation_vector)

# Transform points
transformed_points = []

for point in points:
    point_coords = point[:3]
    transformed_point = apply_rigid_body_transform(point_coords, R, T)
    transformed_points.append(transformed_point)

for idx, transformed_point in enumerate(transformed_points):
    print(texts[idx], transformed_point)
    
    
# Visualize final image with bounding boxes using matplotlib
plt.imshow(cv2.cvtColor(aligned_color_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

