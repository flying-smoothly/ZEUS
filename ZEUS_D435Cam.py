import cv2
import numpy as np
import pyrealsense2 as rs
import json
import uuid
import time
import requests
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

def realsense_pipeline(api_url, secret_key):
    # RealSense 파이프라인 시작
    pipeline = rs.pipeline()
    config = rs.config()

    # 색상과 깊이 스트림 활성화 
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30) 
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

    # 파이프라인 시작
    profile = pipeline.start(config)

    # Spatial 필터 설정
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
            points[i] = [depth * x, depth * y, depth + 0.25] #비커의 반지름 값 추가

        return points

    last_ocr_time = time.time()
    recognized_texts = set()
    pixels = []
    texts = []

    try:
        while True:
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

            # 2초에 한 번씩 OCR 요청 수행
            if time.time() - last_ocr_time >= 2 and len(recognized_texts) < 4:
                # OCR 요청 데이터 생성
                _, encoded_img = cv2.imencode('.jpg', aligned_color_image)
                encoded_img = np.array(encoded_img).tobytes()

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
                for i in ocr_result['images'][0]['fields']:
                    text = i['inferText']

                    # 각 target_text와 비교하여 일치하는 경우만 처리
                    for target_text in target_texts:
                        if target_text.lower().strip() in text.lower().strip():
                            recognized_texts.add(target_text)
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
                                if text in target_texts:
                                    pixels.append(pixel)
                                    texts.append(text)

                                # 경계 상자 그리기
                                pts = np.array([[vertex['x'], vertex['y']] for vertex in bounding_box], np.int32)
                                pts = pts.reshape((-1, 1, 2))
                                cv2.polylines(aligned_color_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

                                # Pillow를 사용해 텍스트 표시
                                x, y = pts[0][0]
                                aligned_color_image = draw_text_with_pillow(aligned_color_image, text, (x, y - 10))

            # 이미지 시각화 (실시간)
            aligned_color_image_resized = cv2.resize(aligned_color_image, (0, 0), fx=0.7, fy=0.7)
            cv2.imshow('figure', aligned_color_image_resized)

            # 네 개의 target_texts가 모두 인식되면 종료
            if len(recognized_texts) == 4:
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

    points = rs2_deproject_pixel_to_point(intrin, pixels)
    result_dict = {}

    for idx, point in enumerate(points):
        result_dict[texts[idx]] = point
        
    # point 값이 [0, 0, 0]인 경우 처음 촬영 과정으로 돌아감
    if all(np.array_equal(point, [0, 0, 0]) for point in points):
        print("[0, 0, 0] 값이 감지되었습니다. 처음 촬영 과정으로 돌아갑니다.")
        return realsense_pipeline(api_url, secret_key)

    return result_dict

# 결과 출력
api_url = 'https://7x3edq0zuo.apigw.ntruss.com/custom/v1/35237/5667ef73dac709effcfdc518a98ea9238365ec62b0eb5f478392e28818a2efd5/general'
secret_key = 'eUx4Ulp3TnNOS1daSG9veVpGZEtybmRoalpMZlZQeFY='

results = realsense_pipeline(api_url, secret_key)
TT, RE, V, H = None, None, None, None
for key, value in results.items():
    print(f"{key} : {value}")
    if key == '티트리':
        TT = value
        print(TT)
    elif key == "레티놀":
        RE = value
    elif key == "비타민":
        V = value
    elif key == "히알":
        H = value

list_points = [TT, RE, V, H]
  
camera_coords = np.array([
    [-0.33112039, -0.07665451, 0.84900003],
    [-0.32333284, -0.0347358, 0.91400003],
    [-0.29754899, 0.06729869, 0.91500003],
    [-0.26263612, 0.07601608, 0.90200003],
    [-0.15897727, 0.07243216, 0.95300003],
    [-0.11128693, 0.08555239, 0.96800003],
    [-0.06938264, 0.08249239, 1.00500004],
    [-0.07011815, 0.03528218, 1.00400004],
    [-0.0397538, 0.0542954, 1.02500004],
    [-0.06928613, 0.03500604, 1.01000004],
    [-0.15112403, 0.05614746, 0.97900003],
    [-0.22166276, 0.10795134, 0.95200003],
    [-0.27315097, 0.1059158, 0.95300003],
    [-0.29907427, 0.10475901, 1.00900004],
    [-0.32526207, 0.07344075, 1.10400004],
    [-0.3274068, 0.00877496, 1.08400004],
    [-0.34117389, -0.00974894, 1.01500004],
    [-0.35330308, 0.02079982, 0.99000004],
    [-0.09168035, 0.01827836, 0.97600003],
    [-0.03368649, 0.06828807, 0.92600003],
    [-0.0326974, 0.09800896, 0.92600003],
    [-0.03244104, 0.10472202, 0.93100003],
    [-0.10163861, 0.10555926, 0.93400003],
    [-0.1421249, 0.10702659, 0.92900003],
    [-0.17185864, 0.10895331, 0.95100003],
    [-0.06277156, 0.10763263, 0.92500003],
    [-0.03388962, 0.09847414, 0.90600003],
    [-0.03512749, 0.07926875, 0.88500003],
    [-0.02959944, 0.05877141, 0.88600003],
    [-0.10935769, 0.03025288, 0.89400003]
])

# Updated robot coordinates
robot_coords = np.array([
    [155.66, 435.72, 218.37],
    [225.66, 435.72, 178.37],
    [224.31, 411.45, 77.40],
    [216.27, 377.89, 57.95],
    [276.27, 277.88, 67.95],
    [287.86, 229.99, 54.09],
    [327.86, 189.99, 84.09],
    [297.86, 109.99, 94.10],
    [347.86, 159.99, 84.11],
    [337.86, 189.99, 104.11],
    [297.86, 269.99, 84.12],
    [267.85, 339.98, 34.13],
    [264.85, 389.98, 37.13],
    [324.85, 419.98, 37.14],
    [419.85, 449.98, 67.14],
    [399.85, 449.97, 132.16],
    [329.85, 459.97, 152.16],
    [299.85, 469.97, 122.16],
    [296.85, 209.96, 122.17],
    [246.85, 149.96, 72.18],
    [246.85, 149.96, 42.19],
    [256.85, 149.96, 34.19],
    [256.85, 219.96, 34.19],
    [246.85, 259.96, 34.19],
    [266.85, 289.96, 32.20],
    [246.85, 179.96, 32.20],
    [226.85, 149.96, 42.21],
    [206.85, 149.96, 62.22],
    [210.85, 144.96, 82.22],
    [213.85, 224.96, 112.23]
])

# least square fitting을 통한 A,B matrix 구하기
A = []
B_x, B_y, B_z = [], [], []

# Populate A and B with 30 pairs
for i in range(30):
    x1, y1, z1 = camera_coords[i]
    x2, y2, z2 = robot_coords[i]
    A.append([x1, y1, z1, 1])
    B_x.append(x2)
    B_y.append(y2)
    B_z.append(z2)

A = np.array(A)
B_x = np.array(B_x)
B_y = np.array(B_y)
B_z = np.array(B_z)

# Perform least squares fitting to find transformation parameters
params_x, _, _, _ = np.linalg.lstsq(A, B_x, rcond=None)
params_y, _, _, _ = np.linalg.lstsq(A, B_y, rcond=None)
params_z, _, _, _ = np.linalg.lstsq(A, B_z, rcond=None)

# Function to transform a point using the fitted parameters
def transform_point(x1, y1, z1, params_x, params_y, params_z):
    x2 = params_x[0] * x1 + params_x[1] * y1 + params_x[2] * z1 + params_x[3]
    y2 = params_y[0] * x1 + params_y[1] * y1 + params_y[2] * z1 + params_y[3]
    z2 = params_z[0] * x1 + params_z[1] * y1 + params_z[2] * z1 + params_z[3]
    return x2, y2, z2

# 모든 카메라 좌표를 로봇 좌표로 변환
for i, camera_point in enumerate(list_points):
    if camera_point is not None:
        x1, y1, z1 = camera_point
        transformed_coords = transform_point(x1, y1, z1, params_x, params_y, params_z)
        print("Transformed coordinates:", transformed_coords)