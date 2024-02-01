import os
import pandas as pd
from PIL import Image
from io import BytesIO
import requests
import jsonlines
from multiprocessing import Pool
from tqdm import tqdm
import time
from pathlib import Path

def download_image(data):
    idx, row = data
    image_url, save_dir, caption = row
    image_filename = f"{idx+1:06d}.png"
    image_path = os.path.join(save_dir, image_filename)

    try:
        response = requests.get(image_url)
        time.sleep(0.1)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.save(image_path)
        return {"file_name": image_filename, "caption": caption}
    except Exception as e:
        return None

tsv_files = [
    # ("dataset/Validation_GCC-1.1.0-Validation.tsv", "dataset/valid"),
    ("dataset/Train_GCC-training.tsv", "dataset/train")
]

offset = 150000
num_images = 150000 # 처리할 이미지 수
num_workers = 256  # 병렬 작업에 사용할 프로세스 수

for tsv_file, save_dir in tsv_files:
    df = pd.read_csv(tsv_file, sep='\t', header=None, names=['caption', 'image_url'])
    df = df.iloc[offset:].head(num_images)

    # 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    metadata_file_path = os.path.join("dataset", f"{Path(save_dir).stem}_metadata2.jsonl")
    tasks = [(idx, (row['image_url'], save_dir, row['caption'])) for idx, row in df.iterrows()]

    flush_interval = 200  # 파일 버퍼를 flush할 작업 수

    # multiprocessing.Pool과 jsonlines.Writer를 사용하여 작업 수행
    with Pool(num_workers) as pool:
        f = open(metadata_file_path, 'w')
        with jsonlines.Writer(f) as writer:
            with tqdm(total=len(tasks), desc=f"Downloading {save_dir}") as pbar:
                for i, result in enumerate(pool.imap_unordered(download_image, tasks)):
                    if result:
                        writer.write(result)
                        # 일정 주기마다 flush() 호출
                        if (i + 1) % flush_interval == 0:
                            f.flush()
                    pbar.update(1)
        f.close()
print("All images have been processed.")