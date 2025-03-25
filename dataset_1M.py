import os
import subprocess
import pandas as pd
import random
import shutil
import argparse
import json
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def check_video_resolution(video_path):
    """
    Check if a video has at least 720p resolution using ffprobe (preferred) or OpenCV (fallback).
    
    Args:
        video_path: Path to the video file
        
    Returns:
        tuple: (is_hd, width, height) where is_hd is True if resolution is at least 720p,
               or (False, 0, 0) if video is corrupted or unreadable
    """
    # Try ffprobe first (more reliable for metadata)
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
             '-show_entries', 'stream=width,height', '-of', 'json', video_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10
        )
        data = json.loads(result.stdout)
        width = data['streams'][0]['width']
        height = data['streams'][0]['height']
        is_hd = height >= 720
        print(f"Video {video_path}: Resolution {width}x{height} (via ffprobe), HD: {is_hd}")
        return is_hd, width, height
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, KeyError, json.JSONDecodeError) as e:
        print(f"ffprobe failed for {video_path}: {e}. Falling back to OpenCV.")

    # Fallback to OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video file with OpenCV: {video_path}")
            return False, 0, 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ret, _ = cap.read()
        cap.release()
        if not ret:
            print(f"Could not decode first frame of {video_path}")
            return False, 0, 0
        is_hd = height >= 720
        print(f"Video {video_path}: Resolution {width}x{height} (via OpenCV), HD: {is_hd}")
        return is_hd, width, height
    except Exception as e:
        print(f"Error checking resolution for {video_path}: {e}")
        return False, 0, 0

def download_sample(output_directory, zip_part=0, sample_size=100, test_split=0.15, val_split=0.15, 
                    min_hd=False, keep_existing=False, max_attempts=20):
    """
    Download a single OpenVid-1M ZIP file and extract a random sample of videos.
    
    Args:
        output_directory: Base directory to store all files
        zip_part: Which part (0-185) to download (default: 0)
        sample_size: Number of video-text pairs to download (default: 100)
        test_split: Fraction of data for test set (default: 0.15)
        val_split: Fraction of data for validation set (default: 0.15)
        min_hd: If True, only keep videos with at least 720p resolution (default: False)
        keep_existing: If True, use existing videos and download only what's needed (default: False)
        max_attempts: Maximum number of attempts to find videos (default: 20)
    """
    # Directory setup
    zip_folder = os.path.join(output_directory, "download")
    data_folder = os.path.join(output_directory, "data")
    mapping_folder = os.path.join(data_folder, "mapping")
    train_folder = os.path.join(output_directory, "train")
    test_folder = os.path.join(output_directory, "test")
    val_folder = os.path.join(output_directory, "val")
    low_res_db_path = os.path.join(data_folder, "low_resolution_videos.csv")
    
    for folder in [zip_folder, data_folder, mapping_folder, train_folder, test_folder, val_folder]:
        os.makedirs(folder, exist_ok=True)

    # Load or initialize low-resolution database
    current_zip_low_res = set()
    if os.path.exists(low_res_db_path):
        try:
            low_res_df = pd.read_csv(low_res_db_path)
            current_zip_df = low_res_df[low_res_df['zip_part'] == zip_part]
            current_zip_low_res = set(current_zip_df['video'].tolist())
            print(f"Loaded {len(current_zip_low_res)} known low-res videos for ZIP part {zip_part}")
        except Exception as e:
            print(f"Error loading low-res database: {e}")

    # Download metadata
    metadata_path = os.path.join(data_folder, "OpenVid-1M.csv")
    if not os.path.exists(metadata_path):
        print("Downloading metadata...")
        subprocess.run(["wget", "-O", metadata_path, 
                        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv"], 
                       check=True)
    metadata = pd.read_csv(metadata_path)

    # Download mapping file
    mapping_path = hf_hub_download(
        repo_id="phil329/OpenVid-1M-mapping",
        filename=f"video_mappings/OpenVid_part{zip_part}.csv",
        repo_type="dataset",
        cache_dir=mapping_folder
    )
    mapping_df = pd.read_csv(mapping_path)

    # Check existing videos
    all_existing_videos = set()
    for split_name in ['train', 'test', 'val']:
        split_folder = os.path.join(output_directory, split_name)
        metadata_path = os.path.join(split_folder, f"{split_name}_metadata.csv")
        videos_folder = os.path.join(split_folder, "videos")
        
        if os.path.exists(metadata_path):
            try:
                existing_df = pd.read_csv(metadata_path)
                for video in existing_df['video'].tolist():
                    video_path = os.path.join(videos_folder, video)
                    if os.path.exists(video_path):
                        all_existing_videos.add(video)
            except Exception as e:
                print(f"Error reading {split_name} metadata: {e}")

    print(f"Total existing videos with both metadata and video files: {len(all_existing_videos)}")

    # Filter Potential Videos
    zip_videos = [v for v in mapping_df['video'].tolist() if v not in all_existing_videos]
    print(f"Found {len(zip_videos)} potential videos in ZIP part {zip_part} after filtering existing videos")

    # Additional filtering for low-resolution videos
    zip_videos = [v for v in zip_videos if v not in current_zip_low_res]
    print(f"Found {len(zip_videos)} potential videos after filtering known low-res videos")

    # Determine how many videos to download
    if keep_existing:
        remaining_videos_needed = max(sample_size - len(all_existing_videos), 0)
        if remaining_videos_needed == 0:
            print(f"Already have {len(all_existing_videos)} videos, meeting target of {sample_size}.")
            print("No new downloads needed. Use --keep_existing=False to force a new batch.")
            return
    else:
        remaining_videos_needed = sample_size
        print("Ignoring existing videos for a fresh batch.")

    print(f"Need {remaining_videos_needed} more videos")

    # Download ZIP file
    zip_path = os.path.join(zip_folder, f"OpenVid_part{zip_part}.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading ZIP part {zip_part}...")
        try:
            subprocess.run(["wget", "-O", zip_path, 
                            f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{zip_part}.zip"], 
                           check=True)
        except subprocess.CalledProcessError:
            print("Direct download failed. Trying split parts...")
            part_files = []
            for suffix in ['partaa', 'partab']:
                part_url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{zip_part}_{suffix}"
                part_path = os.path.join(zip_folder, f"OpenVid_part{zip_part}_{suffix}")
                try:
                    subprocess.run(["wget", "-O", part_path, part_url], check=True)
                    part_files.append(part_path)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to download {part_path}: {e}")
            if len(part_files) == 2:
                with open(zip_path, 'wb') as outfile:
                    for part_file in part_files:
                        with open(part_file, 'rb') as infile:
                            shutil.copyfileobj(infile, outfile)
                print("ZIP parts concatenated.")

    # Extract videos
    temp_extract_folder = os.path.join(output_directory, "temp_extract")
    os.makedirs(temp_extract_folder, exist_ok=True)
    successful_hd_videos = []
    attempted_videos = set()
    new_low_res_videos = []
    attempt_count = 0
    batch_size = min(remaining_videos_needed, 1000)

    while (len(successful_hd_videos) < remaining_videos_needed and 
           len(attempted_videos) < len(zip_videos) and 
           attempt_count < max_attempts):
        remaining_videos = [v for v in zip_videos if v not in attempted_videos]
        current_batch_size = min(batch_size, len(remaining_videos))
        if current_batch_size == 0:
            print("No more videos to try.")
            break
        current_batch = random.sample(remaining_videos, current_batch_size)
        attempted_videos.update(current_batch)
        attempt_count += 1
        print(f"Attempt {attempt_count}/{max_attempts}: Extracting {len(current_batch)} videos...")

        for video in tqdm(current_batch):
            video_in_zip = mapping_df[mapping_df['video'] == video]['video_path'].values[0]
            try:
                subprocess.run(["unzip", "-j", zip_path, video_in_zip, "-d", temp_extract_folder], 
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                extracted_path = os.path.join(temp_extract_folder, video)
                if os.path.getsize(extracted_path) < 10000:
                    print(f"Skipping {video}: File too small")
                    os.remove(extracted_path)
                    continue
            except Exception as e:
                print(f"Extraction failed for {video}: {e}")
                continue

        extracted_videos = [f for f in os.listdir(temp_extract_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
        print(f"Extracted {len(extracted_videos)} videos this batch")

        if min_hd:
            for video in extracted_videos:
                video_path = os.path.join(temp_extract_folder, video)
                is_hd, _, _ = check_video_resolution(video_path)
                if is_hd:
                    successful_hd_videos.append(video)
                else:
                    new_low_res_videos.append(video)
                    os.remove(video_path)
        else:
            successful_hd_videos.extend(extracted_videos)

        print(f"Progress: {len(successful_hd_videos)}/{remaining_videos_needed} videos")

    # Update low-res database
    if min_hd and new_low_res_videos:
        new_low_res_df = pd.DataFrame({'zip_part': [zip_part] * len(new_low_res_videos), 'video': new_low_res_videos})
        if os.path.exists(low_res_db_path):
            low_res_df = pd.read_csv(low_res_db_path)
            combined_df = pd.concat([low_res_df, new_low_res_df]).drop_duplicates(subset=['zip_part', 'video'])
            combined_df.to_csv(low_res_db_path, index=False)
        else:
            new_low_res_df.to_csv(low_res_db_path, index=False)
        print(f"Updated low-res database with {len(new_low_res_videos)} entries")

    # Split and save videos
    selected_metadata = metadata[metadata['video'].isin(successful_hd_videos)]
    random.shuffle(successful_hd_videos)
    test_count = int(len(successful_hd_videos) * test_split)
    val_count = int(len(successful_hd_videos) * val_split)
    test_videos = successful_hd_videos[:test_count]
    val_videos = successful_hd_videos[test_count:test_count + val_count]
    train_videos = successful_hd_videos[test_count + val_count:]

    splits = {'train': (train_folder, train_videos), 'test': (test_folder, test_videos), 'val': (val_folder, val_videos)}
    for split_name, (folder, videos) in splits.items():
        split_metadata = selected_metadata[selected_metadata['video'].isin(videos)]
        metadata_path = os.path.join(folder, f"{split_name}_metadata.csv")
        if os.path.exists(metadata_path):
            existing_df = pd.read_csv(metadata_path)
            combined_df = pd.concat([existing_df, split_metadata]).drop_duplicates(subset=['video'])
            combined_df.to_csv(metadata_path, index=False)
        else:
            split_metadata.to_csv(metadata_path, index=False)

        videos_folder = os.path.join(folder, "videos")
        os.makedirs(videos_folder, exist_ok=True)
        for video in videos:
            src = os.path.join(temp_extract_folder, video)
            dst = os.path.join(videos_folder, video)
            if os.path.exists(src):
                try:
                    shutil.move(src, dst)
                except Exception as e:
                    print(f"Failed to move {video}: {e}")
            txt_path = os.path.join(videos_folder, os.path.splitext(video)[0] + '.txt')
            if not os.path.exists(txt_path):
                caption = split_metadata[split_metadata['video'] == video]['caption'].values[0]
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption)
                except Exception as e:
                    print(f"Failed to write caption for {video}: {e}")

    shutil.rmtree(temp_extract_folder)
    print(f"Completed: {len(train_videos)} train, {len(test_videos)} test, {len(val_videos)} val videos")

# Argument parsing remains unchanged
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download videos from a single OpenVid-1M ZIP part')
    parser.add_argument('--output_directory', type=str, default="./data", help="Output directory for the dataset")
    parser.add_argument('--zip_part', type=int, default=0, help="ZIP part number to download")
    parser.add_argument('--sample_size', type=int, default=100, help="Number of video-text pairs to download")
    parser.add_argument('--test_split', type=float, default=0.15, help="Fraction of data for test set")
    parser.add_argument('--val_split', type=float, default=0.15, help="Fraction of data for validation set")
    parser.add_argument('--no_hd_filter', action='store_false', dest='min_hd', default=True, help="Don't filter videos by resolution")
    parser.add_argument('--keep_existing', action='store_true', default=False, help="Use existing videos and download only what's needed")
    parser.add_argument('--max_attempts', type=int, default=20, help="Maximum number of attempts to find videos")
    args = parser.parse_args()
    
    download_sample(args.output_directory, args.zip_part, args.sample_size, args.test_split, 
                    args.val_split, args.min_hd, args.keep_existing, args.max_attempts)