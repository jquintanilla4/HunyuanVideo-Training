import os
import cv2
from collections import Counter
from fractions import Fraction
import glob
from pathlib import Path

def get_video_dimensions(video_path):
    """Get the dimensions of a video file using OpenCV."""
    try:
        # Open the video file
        video = cv2.VideoCapture(video_path)
        
        # Check if video opened successfully
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
            
        # Get video width and height
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Release the video object
        video.release()
        
        return width, height
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def get_aspect_ratio(width, height):
    """Calculate the aspect ratio in simplified form."""
    if width == 0 or height == 0:
        return None
        
    # Ensure we have proper integer values
    width, height = int(width), int(height)
    
    # Simplify the ratio using Fraction
    fraction = Fraction(width, height).limit_denominator(100)
    ratio_float = width / height
    
    # Check for common aspect ratios and round them
    # Standard ratios
    if abs(ratio_float - 16/9) < 0.01:
        return "16:9"  # Standard widescreen
    elif abs(ratio_float - 4/3) < 0.01:
        return "4:3"   # Standard fullscreen
    elif abs(ratio_float - 1) < 0.01:
        return "1:1"   # Square
    elif abs(ratio_float - 9/16) < 0.01:
        return "9:16"  # Vertical video
        
    # Cinematic ratios
    elif abs(ratio_float - 2.35) < 0.05:
        return "2.35:1"  # Cinemascope/Anamorphic
    elif abs(ratio_float - 2.39) < 0.05:
        return "2.39:1"  # Modern anamorphic widescreen
    elif abs(ratio_float - 2.40) < 0.05:
        return "2.40:1"  # Modern theatrical widescreen
    elif abs(ratio_float - 1.85) < 0.05:
        return "1.85:1"  # Standard theatrical widescreen
    elif abs(ratio_float - 2.20) < 0.05:
        return "2.20:1"  # 70mm IMAX
    elif abs(ratio_float - 1.78) < 0.02:
        return "16:9"    # Exactly 1.78:1 is still 16:9
    elif abs(ratio_float - 1.33) < 0.02:
        return "4:3"     # Exactly 1.33:1 is still 4:3
    
    # Otherwise return the calculated ratio
    return f"{fraction.numerator}:{fraction.denominator}"

def analyze_videos(base_dir='data'):
    """Analyze videos in train, test, and val folders."""
    folders = ['train', 'test', 'val']
    dimensions_counter = Counter()
    ratios_counter = Counter()
    
    for folder in folders:
        video_dir = os.path.join(base_dir, folder, 'videos')
        if not os.path.exists(video_dir):
            print(f"Directory {video_dir} does not exist, skipping...")
            continue
            
        video_files = glob.glob(os.path.join(video_dir, '**', '*.mp4'), recursive=True)
        print(f"Found {len(video_files)} videos in {video_dir}")
        
        for video_path in video_files:
            dimensions = get_video_dimensions(video_path)
            if dimensions:
                width, height = dimensions
                dimensions_counter[f"{width} x {height}"] += 1
                
                ratio = get_aspect_ratio(width, height)
                if ratio:
                    ratios_counter[ratio] += 1
    
    # Print results
    print("\nVideo Dimensions:")
    for dim, count in sorted(dimensions_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{dim} videos - {count}")
    
    print("\nAspect Ratios:")
    for ratio, count in sorted(ratios_counter.items(), key=lambda x: x[1], reverse=True):
        print(f"{ratio} videos - {count}")
    
    # Overall statistics
    total_videos = sum(dimensions_counter.values())
    print(f"\nTotal videos analyzed: {total_videos}")

if __name__ == "__main__":
    analyze_videos() 