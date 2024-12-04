import sys
from tqdm import tqdm

import csv
import os
import datetime
import random
from utils import *
import cv2
import numpy as np


# OpenCV-based optical flow replacement
def optical_flow(prev_frame, frames):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    flows = []
    for frame in frames:
        next_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flows.append(flow)
        prev_gray = next_gray
    return np.array(flows)


# Function to resize and pad the optical flow
def resize_and_pad_flow(flow, target_shape):
    """
    將光流數據調整為目標形狀。
    :param flow: 原始光流數據 (74, 288, 288, 2)
    :param target_shape: 目標形狀 (149, 160, 256, 2)
    :return: 調整後的光流數據
    """
    target_frames, target_height, target_width, channels = target_shape
    current_frames, current_height, current_width, _ = flow.shape

    # 調整每一張光流圖的大小
    resized_flow = []
    for i in range(current_frames):
        resized_frame = cv2.resize(flow[i], (target_width, target_height))
        resized_flow.append(resized_frame)
    resized_flow = np.array(resized_flow)  # (74, 160, 256, 2)

    # 補齊幀數
    if resized_flow.shape[0] < target_frames:
        padding_frames = target_frames - resized_flow.shape[0]
        padding = np.repeat(resized_flow[-1][np.newaxis, ...], padding_frames, axis=0)
        resized_flow = np.concatenate((resized_flow, padding), axis=0)

    return resized_flow


# Directory and file configuration
base_dir = "C:/Users/user/generative-image-dynamics-main/data/models/unet_v1_samples"
video_dir = base_dir  # 如果所有影片都在這個目錄中
flow_dir = "C:/Users/user/generative-image-dynamics-main/data/flow"
csv_file_name = os.path.join(base_dir, "motion_synthesis_train_set.csv")

num_frames = 75
width = 288
height = 288
train_set = []
train_set_ids = []
target_shape = (149, 160, 256, 2)  # 目標光流形狀


def find_video_files(directory):
    print(f"Searching for videos in: {directory}")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.mp4'):
                video_path = os.path.join(root, file)
                print(f"Found video: {video_path}")
                yield video_path


def parse_one_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 獲取總幀數
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 獲取幀率
    videoid = os.path.basename(video_file).replace(".mp4", "")
    print(f">> video: {videoid} fps: {fps} frames: {frame_count}")

    if fps == 29 or fps == 23:  # 修正某些常見的 fps 值
        fps += 1

    # 固定生成兩個序列，分別從 0 和 5 幀開始
    seqs = [0, 5]
    valid_seqs = []  # 存放有效的序列

    for time_s in seqs:
        print(f"Frame count: {frame_count}, Sequence start: {time_s}, Num frames per sequence: {num_frames}")
        if frame_count < (time_s * fps + num_frames):  # 確保幀數足夠生成序列
            print(f"Skipping sequence {time_s}, not enough frames.")
            continue
        valid_seqs.append(time_s)  # 如果有效，加入有效序列列表

    print(f"Generated valid sequences for {videoid}: {valid_seqs}")
    return videoid, fps, frame_count, valid_seqs


def write_csv():
    fields = ['video_id', 'start_sec', 'num_frames', 'frames_per_sec']
    # 確保目錄存在
    os.makedirs(os.path.dirname(csv_file_name), exist_ok=True)
    # 寫入 CSV 文件
    with open(csv_file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(train_set)


def load_csv(checkvideo_exists=False):
    """
    加載 CSV 文件中的數據，並檢查相關視頻文件是否存在。
    """
    if not os.path.exists(csv_file_name):
        return  # 如果文件不存在，直接返回

    missing_videoids = []  # 存放缺失的視頻ID

    with open(csv_file_name, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            videoid = row["video_id"]
            time_s = int(row["start_sec"])

            if checkvideo_exists:  # 如果需要檢查視頻文件是否存在
                if videoid in missing_videoids:
                    continue  # 如果之前已經標記為缺失，跳過
                videofile = os.path.join(video_dir, videoid + ".mp4")
                if not os.path.exists(videofile):  # 檢查視頻文件是否存在
                    print(f"Found missing video: {videoid}")
                    missing_videoids.append(videoid)
                    continue

            flowfile = os.path.join(flow_dir, f"{videoid}_{time_s:03d}.npy")
            if not os.path.exists(flowfile):  # 如果光流文件不存在，跳過
                continue

            train_set.append(row)  # 添加到數據集

            if videoid not in train_set_ids:  # 如果ID不在已存在的列表中，加入
                train_set_ids.append(videoid)


if __name__ == "__main__":
    load_csv()
    print("previous records:", len(train_set))
    print("skipped:", len(train_set_ids), train_set_ids)

    all_videos = list(find_video_files(video_dir))
    all_videos.sort()
    print("all_videos:", len(all_videos))

    for video in tqdm(all_videos):
        videoid, fps, frame_count, seqs = parse_one_video(video)  # 接收 frame_count
        if videoid is None:
            continue
        if videoid in train_set_ids:
            continue
        for time_s in seqs:
            flow_path = os.path.join(flow_dir, f"{videoid}_{time_s:03d}.npy")
            print(f"Processing sequence: {time_s}, Flow path: {flow_path}")
            if not os.path.exists(flow_path):
                start_frame = int(time_s * fps)  # 起始幀索引
                end_frame = min(start_frame + num_frames, frame_count)  # 調整結束幀索引

                # 如果有效幀範圍不足 2 幀，跳過
                if end_frame - start_frame < 2:
                    print(f"Skipping sequence {time_s}, insufficient frames ({end_frame - start_frame} < 2).")
                    continue

                # 提取幀範圍
                frames = get_frames(video, w=width, h=height, start_sec=time_s, fps=fps, f=(end_frame - start_frame))
                if frames is None or len(frames) == 0:
                    print(f"!! Failed to load frames for {videoid}, time_s: {time_s}.")
                    continue
                else:
                    print(f"Loaded frames for {videoid}, time_s: {time_s}, Frame shape: {frames.shape}")

                # 光流計算與保存
                flow = optical_flow(frames[0], frames[1:])
                resized_flow = resize_and_pad_flow(flow, target_shape)
                os.makedirs(flow_dir, exist_ok=True)  # 確保目錄存在
                save_npy(resized_flow, flow_path, dtype=np.float16)
                print(f"Saved resized flow to: {flow_path}, Shape: {resized_flow.shape}")

            train_set.append(dict(video_id=videoid, start_sec=time_s, num_frames=num_frames, frames_per_sec=fps))
        train_set_ids.append(videoid)
    print("write csv!")
    write_csv()
