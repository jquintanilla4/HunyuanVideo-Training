import requests
import os
import time
import inquirer
import json
from tqdm import tqdm

def load_video_database(output_dir):
    """
    Load the video database from a JSON file or create a new one if it doesn't exist
    
    Args:
        output_dir (str): Directory where the database file is stored
        
    Returns:
        tuple: (database dict, set of existing video IDs)
    """
    db_file = os.path.join(output_dir, "downloaded_videos.json")
    
    if os.path.exists(db_file):
        with open(db_file, 'r') as f:
            try:
                downloaded_db = json.load(f)
            except json.JSONDecodeError:
                downloaded_db = {"videos": {}}
    else:
        downloaded_db = {"videos": {}}
    
    # Ensure we have a valid videos dictionary
    if "videos" not in downloaded_db:
        downloaded_db["videos"] = {}
    
    existing_videos = set(downloaded_db["videos"].keys())
    
    # Count actual video files in directory
    mp4_count_pre = len([f for f in os.listdir(output_dir) if f.endswith('.mp4')])
    print(f"Found {mp4_count_pre} existing videos in database")
    
    return downloaded_db, existing_videos


def save_video_to_database(downloaded_db, video_id, video_info, output_dir, force_save=False):
    """
    Add a video entry to the database and save if needed
    
    Args:
        downloaded_db (dict): The database dictionary
        video_id (str): ID of the video
        video_info (dict): Video metadata to save
        output_dir (str): Directory where the database file is stored
        force_save (bool): Whether to force writing to the database file
        
    Returns:
        int: Current count of videos in database
    """
    # Add to database
    downloaded_db["videos"][video_id] = video_info
    
    # Update database file if forced or periodically
    count = len(downloaded_db["videos"])
    if force_save or count % 10 == 0:
        db_file = os.path.join(output_dir, "downloaded_videos.json")
        with open(db_file, 'w') as f:
            json.dump(downloaded_db, f, indent=2)
    
    return count


def find_best_video_file(video_files, preferred_height=720, preferred_width=1280):
    """
    Find the best video file based on resolution preferences
    
    Args:
        video_files (list): List of video file objects from Pexels API
        preferred_height (int): Preferred height in pixels
        preferred_width (int): Preferred width in pixels
        
    Returns:
        dict or None: The selected video file or None if no suitable file found
    """
    selected_video = None
    
    # First, try to find exact match for preferred resolution MP4
    for file in video_files:
        if (file.get("height") == preferred_height and 
            file.get("width") == preferred_width and
            file.get("file_type") == "video/mp4"):
            selected_video = file
            break
    
    # If no exact match, look for closest resolution MP4
    if not selected_video:
        closest_diff = float('inf')
        for file in video_files:
            if file.get("file_type") == "video/mp4":
                height = file.get("height", 0)
                width = file.get("width", 0)
                # Calculate how close this file is to preferred dimensions
                diff = abs(height - preferred_height) + abs(width - preferred_width)
                if diff < closest_diff:
                    selected_video = file
                    closest_diff = diff
    
    # If still no suitable video, use any MP4
    if not selected_video:
        for file in video_files:
            if file.get("file_type") == "video/mp4":
                selected_video = file
                break
    
    return selected_video


def download_popular_videos(api_key, total_videos=200, per_page=80, preferred_height=720, 
                            preferred_width=1280, output_dir="pexels_videos"):
    """
    Download popular videos from Pexels API with specified resolution preferences
    
    Args:
        api_key (str): Your Pexels API key
        total_videos (int): Total number of videos to download
        per_page (int): Number of videos per page (max 80)
        preferred_height (int): Preferred height in pixels
        preferred_width (int): Preferred width in pixels
        output_dir (str): Directory to save videos
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load database
    downloaded_db, existing_videos = load_video_database(output_dir)
    
    headers = {
        "Authorization": api_key
    }
    
    base_url = "https://api.pexels.com/videos/popular"
    
    # Calculate number of pages needed
    num_pages = (total_videos + per_page - 1) // per_page
    
    # Keep track of downloaded videos
    downloaded_count = 0
    skipped_count = 0
    
    for page in range(1, num_pages + 1):
        # Break if we've downloaded enough videos
        if downloaded_count >= total_videos:
            break
        
        # Build request URL
        params = {
            "page": page,
            "per_page": per_page
        }
        
        try:
            # Get list of popular videos
            response = requests.get(base_url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            videos = data.get("videos", [])
            
            if not videos:
                print(f"No videos found on page {page}")
                break
            
            print(f"Processing page {page}/{num_pages} ({len(videos)} videos)")
            
            # Process each video
            for video in videos:
                if downloaded_count >= total_videos:
                    break
                
                video_id = str(video.get("id"))
                
                # Skip if video already exists in database
                if video_id in existing_videos:
                    print(f"Skipping video {video_id} (already downloaded)")
                    skipped_count += 1
                    continue
                
                video_files = video.get("video_files", [])
                
                if not video_files:
                    print(f"No video files found for video {video_id}")
                    continue
                
                # Find best quality video
                selected_video = find_best_video_file(video_files, preferred_height, preferred_width)
                
                if not selected_video:
                    print(f"No suitable video file found for video {video_id}")
                    continue
                
                # Download the video
                video_url = selected_video.get("link")
                width = selected_video.get('width')
                height = selected_video.get('height')
                resolution = f"{width}x{height}"
                video_path = os.path.join(output_dir, f"{video_id}_{resolution}.mp4")
                
                try:
                    print(f"Downloading video {video_id} ({resolution}) ({downloaded_count + 1}/{total_videos})")
                    download_file(video_url, video_path)
                    downloaded_count += 1
                    
                    # Create video info metadata
                    video_info = {
                        "filename": f"{video_id}_{resolution}.mp4",
                        "resolution": resolution,
                        "width": width,
                        "height": height,
                        "download_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "url": video_url
                    }
                    
                    # Add to database
                    save_video_to_database(downloaded_db, video_id, video_info, output_dir)
                    existing_videos.add(video_id)
                    
                    # Add a small delay to be nice to the API
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"Error downloading video {video_id}: {str(e)}")
            
            # Check if there's a next page
            if "next_page" not in data or data["next_page"] is None:
                print("No more pages available")
                break
                
            # Add a delay between pages to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching page {page}: {str(e)}")
            time.sleep(5)  # Wait longer on error
    
    # Final update to database
    db_file = os.path.join(output_dir, "downloaded_videos.json")
    with open(db_file, 'w') as f:
        json.dump(downloaded_db, f, indent=2)
    
    # Count actual video files in directory
    mp4_count_post = len([f for f in os.listdir(output_dir) if f.endswith('.mp4')])
    
    print(f"Downloaded {downloaded_count} new videos to {output_dir}")
    print(f"Skipped {skipped_count} videos that were already downloaded")
    print(f"Total videos in collection: {mp4_count_post}")


def download_file(url, path):
    """Download a file from URL to the specified path with progress bar"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(path, 'wb') as file, tqdm(
        desc=os.path.basename(path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)


if __name__ == "__main__":
    questions = [
        inquirer.Text('api_key', message='Enter your Pexels API key'),
        inquirer.Text('count', message='Number of videos to download', default='200'),
        inquirer.Text('height', message='Preferred video height (pixels)', default='720'),
        inquirer.Text('width', message='Preferred video width (pixels)', default='1280'),
        inquirer.Text('output', message='Output directory', default='data/pexels')
    ]
    
    answers = inquirer.prompt(questions)
    
    download_popular_videos(
        answers['api_key'], 
        int(answers['count']),
        preferred_height=int(answers['height']),
        preferred_width=int(answers['width']), 
        output_dir=answers['output']
    )