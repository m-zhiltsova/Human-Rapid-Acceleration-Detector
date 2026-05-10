import cv2
import numpy as np
import os
import imageio
import imageio_ffmpeg
import argparse

def resize_with_padding(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    padded[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded

def process_photo(input_path, output_path):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл изображения не найден: {input_path}")

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение: {input_path}")

    result = resize_with_padding(img, 1024, 576)
    cv2.imwrite(output_path, result)
    print(f"Фото сохранено: {output_path}")

def process_video(input_path, output_path):
    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_path, fps=fps, format='mp4', codec='libx264')

    for frame in reader:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        processed = resize_with_padding(frame_bgr, 1024, 576)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        writer.append_data(processed_rgb)

    writer.close()
    print(f"Видео сохранено: {output_path} (fps={fps})")

def convert_media(photo_path, video_path, output_photo_path=None, output_video_path=None):
    if output_photo_path is None:
        base, ext = os.path.splitext(photo_path)
        output_photo_path = f"{base}_new{ext}"
    if output_video_path is None:
        base, ext = os.path.splitext(video_path)
        output_video_path = f"{base}_new{ext}"

    process_photo(photo_path, output_photo_path)
    process_video(video_path, output_video_path)

    return output_photo_path, output_video_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Конвертер фото и видео в размер 1024x576 с чёрными полями')
    parser.add_argument('--photo_path', help='Путь к исходному изображению')
    parser.add_argument('--video_path', help='Путь к исходному видео')
    parser.add_argument('--output-photo', dest='output_photo', default=None,
                        help='Путь для сохранения обработанного фото (по умолчанию: исходное_имя_resized.jpg)')
    parser.add_argument('--output-video', dest='output_video', default=None,
                        help='Путь для сохранения обработанного видео (по умолчанию: исходное_имя_resized.mp4)')

    args = parser.parse_args()

    try:
        out_photo, out_video = convert_media(args.photo_path, args.video_path,
                                             args.output_photo, args.output_video)
        print("Готово!")
    except Exception as e:
        print(f"Ошибка: {e}")
