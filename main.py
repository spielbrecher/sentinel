import json

import customtkinter
from ultralytics import YOLO, checks, hub
import os
import cv2
import csv
import app
import requests
from fastapi import FastAPI, File, UploadFile


def train():
    hub.login('dbdc07d300d183843023fd45f46ecb1aab90c7749c')
    model = YOLO('https://hub.ultralytics.com/models/R9DyTSbdPArqhYrIMiUi')
    model.train()


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


def button_function():
    submission = "result.csv"
    with open(submission, mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=";", lineterminator="\r")
        file_writer.writerow(["filename", "cases_count", "timestamps"])
        # f = os.path.join(directory, "1.mp4")
        decision = recognize("H:\\Programmer\\2023\\sentinel\\videos\\1.mp4")
        print(decision)
        file_writer.writerow(decision)


def start():
    customtkinter.set_appearance_mode("System")  # Modes: system (default), light, dark
    customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

    app = customtkinter.CTk()  # create CTk window like you do with the Tk window
    app.geometry("1024x768")
    # Buttons
    buttonLoad = customtkinter.CTkButton(master=app, text="Загрузить", command=button_function)
    buttonLoad.place(relx=0.1, rely=0.1, anchor=customtkinter.CENTER)

    app.mainloop()


if __name__ == '__main__':
    file = "./videos/1.mp4"
    response = requests.get(url='http://127.0.0.1:8000/hello', data=json.dumps({"value": "Andy"}))
    print('response ->')
    answer = dict(response.json())
    print(answer)

    #button_function()


