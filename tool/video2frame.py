import cv2
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm


input_dir = '/data/UCF101'
output_dir = '/data/ucf101_image'

resize_height = 182
resize_width = 242

print('Please wait! Processing Begins.')


def preprocess():
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, 'train'))
        os.mkdir(os.path.join(output_dir, 'val'))
        os.mkdir(os.path.join(output_dir, 'test'))

    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        video_files = [name for name in os.listdir(file_path)]

        train_and_vilid, test = train_test_split(video_files, test_size=0.2, random_state=42)
        train, val = train_test_split(train_and_vilid, test_size=0.2, random_state=42)

        train_dir = os.path.join(output_dir, 'train', file)
        val_dir = os.path.join(output_dir, 'val', file)
        test_dir = os.path.join(output_dir, 'test', file)

        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        print("The train video porcess...")
        for video in tqdm(train):
            process_video(video, file, train_dir)

        print("The val video porcess...")
        for video in tqdm(val):
            process_video(video, file, val_dir)

        print("The test video porcess...")
        for video in tqdm(test):
            process_video(video, file, test_dir)

    print('Congratulations! Processing Ends.')


def process_video(video, file, save_dir):
    video_filename = video.split('.')[0]
    if not os.path.exists(os.path.join(save_dir, video_filename)):
        os.mkdir(os.path.join(save_dir, video_filename))

    capture = cv2.VideoCapture(os.path.join(input_dir, file, video))

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    extraction_interval = 4
    while (frame_count // extraction_interval <= 16 and extraction_interval > 0):
        extraction_interval -= 1

    count = 0
    success = True
    i = 0

    while (count < frame_count and success):
        success, frame = capture.read()
        if frame is None:
            continue
        if count % extraction_interval == 0:
            if (frame_width != resize_width) or (frame_height != resize_height):
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(filename=os.path.join(save_dir, video_filename, '000{}.jpg'.format(str(i))), img=frame)
            i += 1
        count += 1
    capture.release()


if __name__ == '__main__':
    preprocess()
