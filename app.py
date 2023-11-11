import requests
from ultralytics import YOLO, checks, hub
import os
import cv2
import csv
import ssl
from fastapi import FastAPI, File, UploadFile
import shutil

# uvicorn app:app --host 127.0.0.1 --port 8000
app = FastAPI()

@app.get("/hello")
def get_hello(name):
    n = name.get("value")
    return {"answer": name}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # Получаем файл из запроса
    file_name = file.filename
    with open("test.mp4", 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    # recognize uploaded file
    result = recognize("test.mp4")
    return {"result": result}


def recognize(filename):
    cv2.WINDOW_AUTOSIZE = True
    cv2.setUseOptimized(True)
    VIDEOS_DIR = os.path.join('.', 'videos')
    video_path = os.path.join(VIDEOS_DIR, filename)
    video_path_out = '{}_out.mp4'.format(video_path)

    cap = cv2.VideoCapture(filename)
    ret, frame = cap.read()
    H, W, _ = frame.shape
    out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

    model_path = os.path.join('.', 'weapon1', 'weights', 'weapon_yolo8.v1i.pt')

    # Load a model
    model = YOLO(model_path)  # load a custom model

    threshold = 0.2
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("--- fps --- {}".format(fps))
    cases = 0
    cur_frame = 0
    cur_hour = 0
    cur_min = 0
    cur_sec = 0
    cur_frame_buffer = 0
    timestamps = list()
    last_frame_detected = 0

    while ret:
        # work with time
        cur_frame += 1
        cur_frame_buffer += 1
        if (cur_frame_buffer >= fps):
            cur_sec += 1
            cur_frame_buffer = 0

        if (cur_sec > 59):
            cur_min += 1
            cur_sec = 0
        if (cur_min > 59):
            cur_min = 0
            cur_hour += 1

        if cur_frame % fps != 0:
            out.write(frame)
            ret, frame = cap.read()
            continue


        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if (score > threshold):
                # Detected
                last_frame_detected = cur_frame
                cases += 1
                timestamps.append("{}:{}".format(cur_min, cur_sec))

                print("---------------------------------------------------------------------------------")
                print("frame {} ==== time {}:{} ====".format(cur_frame, cur_min, cur_sec))
                print(result)
                print("---------------------------------------------------------------------------------")
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        out.write(frame)
        ret, frame = cap.read()

    # decision format
    decision = list()
    decision.append(filename)
    decision.append(cases)
    decision.append(timestamps)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return decision


def process_video(video_file):
    # Открываем видеофайл
    video = cv2.VideoCapture(video_file)

    # Получаем частоту кадров
    fps = video.get(cv2.CAP_PROP_FPS)

    # Инициализируем счетчик кадров
    frame_counter = 0

    # Цикл по кадрам видео
    while True:
        # Получаем текущий кадр
        ret, frame = video.read()

        # Если кадр не был получен, выходим из цикла
        if not ret:
            break

        # Обрабатываем кадр
        # ...

        # Увеличиваем счетчик кадров
        frame_counter += 1

    # Закрываем видеофайл
    video.release()

    return frame_counter
