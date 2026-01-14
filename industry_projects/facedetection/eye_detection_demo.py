#!/usr/bin/env python3
"""
Eye Detection Demo Script
This script demonstrates different usage examples of the eye detection system.
"""

import subprocess
import sys
import os

def run_demo():
    """Run demonstration of different eye detection modes"""
    
    print("=== Eye Openness Detection Demo ===\n")
    
    # Check if required files exist
    if not os.path.exists('eye_openness.py'):
        print("Error: eye_openness.py not found!")
        return
    
    if not os.path.exists('eye_detection_script.py'):
        print("Error: eye_detection_script.py not found!")
        return
    
    print("Available demo modes:")
    print("1. Camera/Webcam detection (real-time)")
    print("2. Image processing")
    print("3. Video processing")
    print("4. Show usage examples")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\nSelect mode (1-5): ").strip()
            
            if choice == '1':
                print("\nStarting camera detection...")
                print("Note: This will open your webcam. Press 'q' to quit.")
                input("Press Enter to continue or Ctrl+C to cancel...")
                try:
                    subprocess.run([sys.executable, 'eye_detection_script.py', '--mode', 'camera'])
                except KeyboardInterrupt:
                    print("Camera detection cancelled.")
            
            elif choice == '2':
                # List available images
                image_files = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    print("\nAvailable images:")
                    for i, img in enumerate(image_files, 1):
                        print(f"  {i}. {img}")
                    
                    try:
                        img_choice = int(input(f"Select image (1-{len(image_files)}): ")) - 1
                        if 0 <= img_choice < len(image_files):
                            selected_image = image_files[img_choice]
                            print(f"\nProcessing image: {selected_image}")
                            subprocess.run([sys.executable, 'eye_detection_script.py', 
                                          '--mode', 'image', '--input', selected_image])
                        else:
                            print("Invalid selection!")
                    except ValueError:
                        print("Invalid input!")
                else:
                    print("No image files found in current directory.")
                    custom_path = input("Enter image path (or press Enter to skip): ").strip()
                    if custom_path and os.path.exists(custom_path):
                        subprocess.run([sys.executable, 'eye_detection_script.py', 
                                      '--mode', 'image', '--input', custom_path])
            
            elif choice == '3':
                # List available video files
                video_files = [f for f in os.listdir('.') if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                if video_files:
                    print("\nAvailable videos:")
                    for i, vid in enumerate(video_files, 1):
                        print(f"  {i}. {vid}")
                    
                    try:
                        vid_choice = int(input(f"Select video (1-{len(video_files)}): ")) - 1
                        if 0 <= vid_choice < len(video_files):
                            selected_video = video_files[vid_choice]
                            print(f"\nProcessing video: {selected_video}")
                            subprocess.run([sys.executable, 'eye_detection_script.py', 
                                          '--mode', 'video', '--input', selected_video])
                        else:
                            print("Invalid selection!")
                    except ValueError:
                        print("Invalid input!")
                else:
                    print("No video files found in current directory.")
                    custom_path = input("Enter video path (or press Enter to skip): ").strip()
                    if custom_path and os.path.exists(custom_path):
                        subprocess.run([sys.executable, 'eye_detection_script.py', 
                                      '--mode', 'video', '--input', custom_path])
            
            elif choice == '4':
                show_usage_examples()
            
            elif choice == '5':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice! Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

def show_usage_examples():
    """Show usage examples for the eye detection script"""
    print("\n=== Usage Examples ===")
    print()
    print("1. Camera/Webcam Detection:")
    print("   python eye_detection_script.py --mode camera")
    print("   python eye_detection_script.py --mode camera --camera-index 1")
    print("   python eye_detection_script.py --mode camera --threshold 0.25")
    print()
    print("2. Image Processing:")
    print("   python eye_detection_script.py --mode image --input photo.jpg")
    print("   python eye_detection_script.py --mode image --input photo.jpg --output result.jpg")
    print()
    print("3. Video Processing:")
    print("   python eye_detection_script.py --mode video --input video.mp4")
    print("   python eye_detection_script.py --mode video --input video.mp4 --output output.mp4")
    print("   python eye_detection_script.py --mode video --input video.mp4 --no-save")
    print()
    print("Parameters:")
    print("  --mode: camera, video, or image")
    print("  --input: input file path (required for video/image)")
    print("  --output: output file path (optional)")
    print("  --threshold: EAR threshold (default: 0.2)")
    print("  --camera-index: camera index (default: 0)")
    print("  --no-save: don't save output video")
    print()
    print("Interactive controls (camera/video modes):")
    print("  'q' - Quit")
    print("  's' - Save current frame (camera mode)")
    print("  'r' - Reset statistics (camera mode)")
    print("  SPACE - Pause/Resume")

if __name__ == "__main__":
    run_demo()
