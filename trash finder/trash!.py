from ultralytics import YOLO
import cv2
import time

model = YOLO(
r"C:\Users\AIcontents\Desktop\2026-Mirae-AI-Research-Team-AI\trash finder\runs\classify\train-3\weights\best.pt"
)

c
apture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not capture.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()



prev_time = 0


def draw_crosshair(frame):
    """중앙 에임 표시"""
    h, w = frame.shape[:2]

    cx = w // 2
    cy = h // 2

    color = (0, 0, 0)

    size = 25
    thickness = 2

  
    cv2.line(
        frame,
        (cx - size, cy),
        (cx + size, cy),
        color,
        thickness
    )

    cv2.line(
        frame,
        (cx, cy - size),
        (cx, cy + size),
        color,
        thickness
    )


    cv2.circle(
        frame,
        (cx, cy),
        5,
        color,
        0
    )


def draw_ui(frame, label, confidence, fps):

    h, w = frame.shape[:2]


    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (300, 130),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.45,
        frame,
        0.55,
        0
    )



    cv2.putText(
        frame,
        "님 쓰레기 같음;;;;;;;;;",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )


    cv2.putText(
        frame,
        label,
        (20,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )


    bar_x = 20
    bar_y = 90
    bar_w = 200
    bar_h = 15


    cv2.rectangle(
        frame,
        (bar_x,bar_y),
        (bar_x+bar_w,bar_y+bar_h),
        (80,80,80),
        -1
    )


    cv2.rectangle(
        frame,
        (bar_x,bar_y),
        (
            bar_x+int(bar_w*confidence),
            bar_y+bar_h
        ),
        (0,255,0),
        -1
    )


    cv2.putText(
        frame,
        f"{confidence*100:.1f}%",
        (230,105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )


    cv2.putText(
        frame,
        f"FPS : {fps:.1f}",
        (500,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )



    cv2.putText(
        frame,
        "ESC : EXIT",
        (500,460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255,255,255),
        1
    )


    return frame



while True:

    ret, frame = capture.read()

    if not ret or frame is None:
        continue


    start = time.time()



    results = model(frame, verbose=False)

    result = results[0]

    probs = result.probs

    class_id = probs.top1

    confidence = probs.top1conf.item()


    label = result.names[class_id]



    current_time = time.time()

    fps = 1 / (current_time - prev_time) if prev_time else 0

    prev_time = current_time




    frame = draw_ui(
        frame,
        label,
        confidence,
        fps
    )


    draw_crosshair(frame)



    cv2.rectangle(
        frame,
        (0,0),
        (639,479),
        (0,0,0),
        2
    )


    cv2.imshow(
        "trash finder",
        frame
    )


    if cv2.waitKey(33) == 27:
        break



capture.release()
cv2.destroyAllWindows()