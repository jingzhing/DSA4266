import cv2
import random as rand
import yt_dlp
import os

def download_video(url):
    ydl_opts = {
    'format': 'bestvideo',
    'quiet': True,
    'merge_output_format': 'mp4',
    'outtmpl': 'yt_dl.%(ext)s',
    }
    video_url = url
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

def get_vid_frames(path):
    vid = cv2.VideoCapture(path)
    frame_val = 0
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_val < total_frames:
        vid.set(cv2.CAP_PROP_POS_FRAMES, frame_val)
        ret, frame = vid.read()
        if ret:
            laplace = cv2.Laplacian(frame, cv2.CV_64F).var()
            if laplace > 7:
                cv2.imwrite(f"./frames/{frame_val}_clear.jpg", frame) ## Saves to frames folder
        frame_val += rand.randint(100, 600)
    vid.release()
    os.remove(path) #  delete file after processing

def download_and_process(url):
    download_video(url)
    print(os.getcwd())
    get_vid_frames("yt_dl.mp4")

