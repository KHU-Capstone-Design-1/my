# my
손목
mport cv2
import mediapipe as mp

# MediaPipe Hands 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# 노트북 카메라를 사용하여 VideoCapture 초기화
cap = cv2.VideoCapture(0)  # 0 또는 1로 변경하여 노트북에 있는 카메라를 선택

# 이전 프레임의 손 위치 초기화
prev_hand_landmarks = None

while cap.isOpened():
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("카메라를 찾을 수 없습니다.")
        break

    # 프레임을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손 감지 수행
    results = hands.process(rgb_frame)

    # 감지된 손이 있을 경우 랜드마크 그리기
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 각 랜드마크의 좌표 얻기
            current_hand_landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

            # 이전 프레임이 있을 경우 손의 움직임 계산
            if prev_hand_landmarks is not None:
                # 손의 움직임을 계산하기 위한 기준 랜드마크 인덱스 (손목 기준)
                wrist_index = 0

                # 현재 손목의 좌표
                cur_wrist_x, cur_wrist_y = current_hand_landmarks[wrist_index]

                # 이전 손목의 좌표
                prev_wrist_x, prev_wrist_y = prev_hand_landmarks[wrist_index]

                # 좌우 움직임 계산
                x_movement = cur_wrist_x - prev_wrist_x

                # 좌회전 또는 우회전 판단
                if x_movement < -0.1:  # 좌회전
                    print("Left turn - 좌회전")
                    # Add your left turn code here

                elif x_movement > 0.1:  # 우회전
                    print("Right turn - 우회전")
                    # Add your right turn code here

            # 현재 랜드마크를 이전 랜드마크로 저장
            prev_hand_landmarks = current_hand_landmarks

            # 각 랜드마크의 좌표 얻기 및 그리기
            for lm_id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

    # 화면에 출력
    cv2.imshow('Hand Tracking', frame)

    # 'ESC' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
