import os
import subprocess
import pandas as pd
import random
import shutil
import argparse
import cv2
from tqdm import tqdm
from huggingface_hub import hf_hub_download

def check_video_resolution(video_path):
    """
    Check if a video has at least 720p resolution using OpenCV.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        tuple: (is_hd, width, height) where is_hd is True if resolution is at least 720p
              or (False, 0, 0) if video is corrupted
    """
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            print(f"Could not open video file: {video_path}")
            return False, 0, 0
        
        # Get width and height
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try to read the first frame to verify the video can be decoded
        ret, frame = cap.read()
        if not ret:
            print(f"Could not decode the first frame of {video_path}")
            cap.release()
            return False, 0, 0
            
        # Get actual frame dimensions (may differ from reported properties)
        actual_height, actual_width = frame.shape[:2]
        if actual_width != width or actual_height != height:
            print(f"Warning: Reported dimensions ({width}x{height}) differ from actual frame dimensions ({actual_width}x{actual_height})")
            width, height = actual_width, actual_height
        
        # Release the video capture object
        cap.release()
        
        # Check if height is at least 720 pixels
        is_hd = height >= 720
        
        print(f"Video {video_path}: Resolution {width}x{height}, HD: {is_hd}")
        
        return is_hd, width, height
        
    except Exception as e:
        print(f"Error checking resolution for {video_path}: {e}")
        return False, 0, 0

def download_sample(output_directory, zip_part=0, sample_size=100, test_split=0.15, val_split=0.15, min_hd=False, keep_existing=False, max_attempts=20):
    """
    Download a single OpenVid-1M ZIP file and extract a random sample of videos.
    
    Args:
        output_directory: Base directory to store all files
        zip_part: Which part (0-185) to download (default: 0)
        sample_size: Number of video-text pairs to download (default: 100)
        test_split: Fraction of data for test set (default: 0.15)
        val_split: Fraction of data for validation set (default: 0.15)
        min_hd: If True, only keep videos with at least 720p resolution (default: True)
        keep_existing: If True, Keep and use existing videos instead of overwriting existing ones with fresh downloads (default: True)
        max_attempts: Maximum number of attempts to find videos (default: 20)
    """
    
    # Create directory structure
    zip_folder = os.path.join(output_directory, "download")
    # Remove the centralized video folder
    data_folder = os.path.join(output_directory, "data")
    mapping_folder = os.path.join(data_folder, "mapping")
    
    # Create splits directories
    train_folder = os.path.join(output_directory, "train")
    test_folder = os.path.join(output_directory, "test")
    val_folder = os.path.join(output_directory, "val")
    
    # Create a database to track low-resolution videos to skip in future runs
    low_res_db_path = os.path.join(data_folder, "low_resolution_videos.csv")
    current_zip_low_res = set()
    
    # Load existing low-resolution video database if it exists
    if os.path.exists(low_res_db_path):
        try:
            low_res_df = pd.read_csv(low_res_db_path)
            # Filter to get only videos for the current zip part
            if not low_res_df.empty and 'zip_part' in low_res_df.columns and 'video' in low_res_df.columns:
                current_zip_df = low_res_df[low_res_df['zip_part'] == zip_part]
                current_zip_low_res = set(current_zip_df['video'].tolist())
            print(f"Loaded {len(current_zip_low_res)} known low-resolution videos for ZIP part {zip_part}")
        except Exception as e:
            print(f"Error loading low-resolution database: {e}")
            current_zip_low_res = set()
    
    for folder in [zip_folder, data_folder, mapping_folder, train_folder, test_folder, val_folder]:
        os.makedirs(folder, exist_ok=True)

    # Download the metadata file
    metadata_url = "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv"
    metadata_path = os.path.join(data_folder, "OpenVid-1M.csv")
    
    if not os.path.exists(metadata_path):
        print(f"Downloading metadata file...")
        subprocess.run(["wget", "-O", metadata_path, metadata_url], check=True)
    
    # Load metadata
    print(f"Loading metadata...")
    metadata = pd.read_csv(metadata_path)
    
    # Download the mapping file for the selected ZIP part
    print(f"Downloading mapping file for part {zip_part}...")
    mapping_path = hf_hub_download(
        repo_id="phil329/OpenVid-1M-mapping",
        filename=f"video_mappings/OpenVid_part{zip_part}.csv",
        repo_type="dataset",
        cache_dir=mapping_folder
    )
    
    # Load the mapping file
    mapping_df = pd.read_csv(mapping_path)
    print(f"Mapping file contains {len(mapping_df)} videos")
    
    # Get all videos in this ZIP file
    zip_videos = mapping_df['video'].tolist()
    
    # Check for existing videos in split folders to avoid duplicate extraction
    existing_train_videos = set()
    existing_test_videos = set()
    existing_val_videos = set()
    
    # Read existing metadata CSVs if they exist
    for split_name, split_folder, existing_set in [
        ('train', train_folder, existing_train_videos),
        ('test', test_folder, existing_test_videos),
        ('val', val_folder, existing_val_videos)
    ]:
        metadata_path = os.path.join(split_folder, f"{split_name}_metadata.csv")
        if os.path.exists(metadata_path):
            try:
                existing_df = pd.read_csv(metadata_path)
                for video in existing_df['video'].tolist():
                    existing_set.add(video)
                print(f"Found {len(existing_set)} existing videos in {split_name} split")
            except Exception as e:
                print(f"Error reading existing {split_name} metadata: {e}")
    
    # Combine all existing videos to avoid extracting duplicates
    all_existing_videos = existing_train_videos.union(existing_test_videos).union(existing_val_videos)
    print(f"Total existing videos across all splits: {len(all_existing_videos)}")
    
    # If keep_existing is False, we'll download a full new batch regardless of existing videos
    if not keep_existing:
        print("Fresh run requested - ignoring existing videos and downloading a full new batch")
        remaining_videos_needed = sample_size
    else:
        remaining_videos_needed = sample_size - len(all_existing_videos)
        
    if remaining_videos_needed <= 0 and not keep_existing:
        print(f"Already have {len(all_existing_videos)} videos, which meets or exceeds the target of {sample_size}.")
        print("No new videos will be downloaded. Use --keep_existing to download a new batch.")
        return
    
    print(f"Need {remaining_videos_needed} more videos to reach target of {sample_size}")
    
    # Download the ZIP file
    zip_url = f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{zip_part}.zip"
    zip_path = os.path.join(zip_folder, f"OpenVid_part{zip_part}.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading ZIP file part {zip_part}...")
        try:
            subprocess.run(["wget", "-O", zip_path, zip_url], check=True)
            print(f"ZIP file downloaded to {zip_path}")
        except subprocess.CalledProcessError as e:
            error_message = f"ZIP file download failed: {e}"
            print(error_message)
            
            # Try alternative download for split files
            print("Attempting to download split parts...")
            part_urls = [
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{zip_part}_partaa",
                f"https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/OpenVid_part{zip_part}_partab"
            ]
            
            part_files = []
            for part_url in part_urls:
                part_file_path = os.path.join(zip_folder, os.path.basename(part_url))
                if not os.path.exists(part_file_path):
                    try:
                        subprocess.run(["wget", "-O", part_file_path, part_url], check=True)
                        print(f"Part file downloaded to {part_file_path}")
                        part_files.append(part_file_path)
                    except subprocess.CalledProcessError as part_e:
                        print(f"Part file download failed: {part_e}")
                else:
                    print(f"Part file {part_file_path} already exists")
                    part_files.append(part_file_path)
            
            # Concatenate parts if all were downloaded
            if len(part_files) == len(part_urls):
                print("Concatenating part files...")
                cat_command = f"cat {' '.join(part_files)} > {zip_path}"
                os.system(cat_command)
                print(f"Combined file created at {zip_path}")
    else:
        print(f"ZIP file {zip_path} already exists")
    
    # Create a temporary extraction directory
    temp_extract_folder = os.path.join(output_directory, "temp_extract")
    os.makedirs(temp_extract_folder, exist_ok=True)
    
    # Modified approach: Keep extracting videos until we have enough HD videos
    successful_hd_videos = []
    
    # Fix the batch size calculation to prevent negative numbers
    if remaining_videos_needed <= 0:
        if not keep_existing:
            print("Already have enough videos. Use --keep_existing=false to force new downloads.")
            shutil.rmtree(temp_extract_folder)
            return
        else:
            # If we want new videos anyway, set a positive batch size
            remaining_videos_needed = sample_size
            
    # Calculate batch size - use full sample size if ≤ 500, otherwise use 1/5 of sample size
    batch_size = remaining_videos_needed if remaining_videos_needed <= 1000 else remaining_videos_needed // 10
    batch_size = min(batch_size, len(zip_videos))  # Ensure we don't exceed available videos

    if batch_size == 0:
        print("No suitable videos available in this ZIP part. Try another ZIP part.")
        if min_hd and len(set(zip_videos).intersection(current_zip_low_res)) > 0:
            print(f"ZIP part {zip_part} has been exhausted of HD (720p+) videos. Please try another ZIP part.")
        shutil.rmtree(temp_extract_folder)
        return
    
    print(f"Starting with batch of {batch_size} videos to try to reach {remaining_videos_needed} HD videos")
    
    # Keep track of videos we've tried
    attempted_videos = set()
    new_low_res_videos = []
    
    attempt_count = 0
    
    while len(successful_hd_videos) < remaining_videos_needed and len(attempted_videos) < len(zip_videos) and attempt_count < max_attempts:
        # Select a batch of videos we haven't tried yet
        remaining_videos = [v for v in zip_videos if v not in attempted_videos]
        current_batch_size = min(batch_size, len(remaining_videos), remaining_videos_needed * 2 - len(successful_hd_videos))
        
        if current_batch_size == 0:
            print("No more videos to try in this ZIP part.")
            break
            
        current_batch = random.sample(remaining_videos, current_batch_size)
        attempted_videos.update(current_batch)
        attempt_count += 1
        
        print(f"Extracting batch of {len(current_batch)} videos (attempt {attempt_count}/{max_attempts})...")
        
        # Extract the current batch
        current_successful = []
        for video in tqdm(current_batch):
            # Find the path of this video within the ZIP
            video_in_zip = mapping_df[mapping_df['video'] == video]['video_path'].values[0]
            
            # Extract to temporary folder
            try:
                # Use a timeout for extraction to handle potential corrupted archives
                extraction_process = subprocess.run(
                    ["unzip", "-j", zip_path, video_in_zip, "-d", temp_extract_folder],
                    check=True,
                    stdout=subprocess.DEVNULL,  # Suppress output
                    stderr=subprocess.DEVNULL,
                    timeout=30  # Set a 30-second timeout for extraction
                )
                
                # The unzip might extract with the original path structure
                # Get the extracted filename
                extracted_name = os.path.basename(video_in_zip)
                extracted_path = os.path.join(temp_extract_folder, extracted_name)
                
                if os.path.exists(extracted_path):
                    # Rename if needed to match the expected video name
                    if extracted_name != video:
                        renamed_path = os.path.join(temp_extract_folder, video)
                        os.rename(extracted_path, renamed_path)
                    
                    # Quick file size check - corrupted videos are often very small
                    file_size = os.path.getsize(os.path.join(temp_extract_folder, video))
                    if file_size < 10000:  # Skip suspiciously small files (less than 10KB)
                        print(f"Skipping suspiciously small file: {video} ({file_size} bytes)")
                        os.remove(os.path.join(temp_extract_folder, video))
                        continue
                    
                    current_successful.append(video)
            
            except subprocess.TimeoutExpired:
                print(f"Extraction timed out for {video} - possibly corrupted file")
            
            except Exception as e:
                print(f"Failed to extract {video}: {e}")
        
        # Filter by resolution if needed
        if min_hd:
            for video in tqdm(current_successful):
                video_path = os.path.join(temp_extract_folder, video)
                is_hd, width, height = check_video_resolution(video_path)
                
                if is_hd:
                    successful_hd_videos.append(video)
                else:
                    new_low_res_videos.append(video)
                    print(f"Removing {video} - resolution too low ({width}x{height})")
                    # Delete low-resolution video from temp folder
                    os.remove(video_path)
            
            print(f"Batch result: {len(successful_hd_videos)}/{remaining_videos_needed} HD videos collected")
        else:
            # If no HD filter, all successful extractions count
            successful_hd_videos.extend(current_successful)
            print(f"Batch result: {len(successful_hd_videos)}/{remaining_videos_needed} videos collected")
        
        # Stop if we have enough videos
        if len(successful_hd_videos) >= remaining_videos_needed:
            print(f"Reached target of {remaining_videos_needed} videos!")
            break
    
    # Update low-resolution database if we're filtering by HD
    if min_hd and new_low_res_videos:
        # Create dataframe for new low-res videos
        new_low_res_df = pd.DataFrame({
            'zip_part': [zip_part] * len(new_low_res_videos),
            'video': new_low_res_videos
        })
        
        # Append to existing or create new
        if os.path.exists(low_res_db_path):
            existing_df = pd.read_csv(low_res_db_path)
            combined_df = pd.concat([existing_df, new_low_res_df], ignore_index=True)
            # Remove potential duplicates
            combined_df = combined_df.drop_duplicates(subset=['zip_part', 'video'])
            combined_df.to_csv(low_res_db_path, index=False)
            print(f"Added {len(new_low_res_videos)} new videos to low-resolution database for ZIP part {zip_part}")
            print(f"Total low-resolution videos in database: {len(combined_df)}")
        else:
            new_low_res_df.to_csv(low_res_db_path, index=False)
            print(f"Created new low-resolution database with {len(new_low_res_videos)} videos")
    
    # Get metadata for successful videos
    selected_metadata = metadata[metadata['video'].isin(successful_hd_videos)]
    
    # Create splits
    print("Creating train/test/val splits...")
    random.shuffle(successful_hd_videos)
    
    # Calculate split sizes
    test_count = int(len(successful_hd_videos) * test_split)
    val_count = int(len(successful_hd_videos) * val_split)
    
    # Split into sets
    test_videos = successful_hd_videos[:test_count]
    val_videos = successful_hd_videos[test_count:test_count+val_count]
    train_videos = successful_hd_videos[test_count+val_count:]
    
    # Create metadata files for each split
    splits = {
        'train': (train_folder, train_videos, existing_train_videos),
        'test': (test_folder, test_videos, existing_test_videos),
        'val': (val_folder, val_videos, existing_val_videos)
    }
    
    # Move videos to respective folders and update CSV files
    for split_name, (folder, videos, existing_videos) in splits.items():
        # Filter metadata for this split
        split_metadata = selected_metadata[selected_metadata['video'].isin(videos)]
        
        # Append to existing metadata CSV or create new one
        split_metadata_path = os.path.join(folder, f"{split_name}_metadata.csv")
        if os.path.exists(split_metadata_path) and len(existing_videos) > 0:
            # Read existing metadata
            existing_split_metadata = pd.read_csv(split_metadata_path)
            # Concatenate with new metadata
            combined_metadata = pd.concat([existing_split_metadata, split_metadata], ignore_index=True)
            # Save combined metadata
            combined_metadata.to_csv(split_metadata_path, index=False)
            print(f"Updated {split_name} metadata: {len(existing_split_metadata)} existing + {len(split_metadata)} new = {len(combined_metadata)} total entries")
        else:
            # Save new metadata
            split_metadata.to_csv(split_metadata_path, index=False)
            print(f"Created new {split_name} metadata with {len(split_metadata)} entries")
        
        # Process videos for this split
        videos_folder = os.path.join(folder, "videos")
        os.makedirs(videos_folder, exist_ok=True)
        
        for video in videos:
            # Move video file from temp folder to split folder
            src = os.path.join(temp_extract_folder, video)
            dst = os.path.join(videos_folder, video)
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.move(src, dst)
            
            # Create matching text file with caption if it doesn't exist
            txt_filename = os.path.splitext(video)[0] + '.txt'
            txt_path = os.path.join(videos_folder, txt_filename)
            
            if not os.path.exists(txt_path):
                # Write caption to text file
                video_caption = split_metadata[split_metadata['video'] == video]['caption'].values[0]
                with open(txt_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(video_caption)
    
    # Remove temporary extraction directory
    shutil.rmtree(temp_extract_folder)
    
    # Print summary
    print("\nDownload and split complete!")

    if min_hd:
        print(f"HD filter: Collected {len(successful_hd_videos)} new videos with at least 720p resolution")
        print(f"Identified {len(new_low_res_videos)} new low-resolution videos (skipped)")
    
    print(f"Train set: {len(train_videos)} new videos")
    print(f"Test set: {len(test_videos)} new videos")
    print(f"Validation set: {len(val_videos)} new videos")
    print(f"Files saved to: {output_directory}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download videos from a single OpenVid-1M ZIP part')
    parser.add_argument('--output_directory', type=str, help='Path to the dataset directory', default="./data")
    parser.add_argument('--zip_part', type=int, help='Which ZIP part to download (0-185) and/or use', default=0)
    parser.add_argument('--sample_size', type=int, help='Number of video-text pairs to download', default=100)
    parser.add_argument('--test_split', type=float, help='Fraction of data for test set', default=0.15)
    parser.add_argument('--val_split', type=float, help='Fraction of data for validation set', default=0.15)
    parser.add_argument('--no_hd_filter', action='store_false', dest='min_hd', default=True, help='Disable HD filtering (at least 720p)')
    parser.add_argument('--keep_existing', action='store_true', dest='keep_existing', default=True, 
                       help='Keep and use existing videos instead of overwriting existing ones with fresh downloads')
    parser.add_argument('--max_attempts', type=int, help='Maximum number of attempts to find videos', default=20)
    args = parser.parse_args()
    
    download_sample(args.output_directory, 
                    args.zip_part,
                    args.sample_size,
                    args.test_split,
                    args.val_split,
                    args.min_hd,
                    args.keep_existing, 
                    args.max_attempts)