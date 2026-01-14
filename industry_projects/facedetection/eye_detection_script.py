#!/usr/bin/env python3
"""
Eye Openness Detection Script
This script demonstrates eye openness detection using MediaPipe for:
1. Real-time webcam detection
2. Video file processing
3. Static image processing

Usage:
    python eye_detection_script.py --mode camera                    # Use webcam
    python eye_detection_script.py --mode video --input video.mp4   # Process video file
    python eye_detection_script.py --mode image --input image.jpg   # Process image
"""

import cv2
import argparse
import os
import sys
from eyelibuz.eye_openness import EyeOpennessDetector


class EyeDetectionApp:
    def __init__(self, ear_threshold=0.15, max_num_faces=5):
        """
        Initialize the Eye Detection Application
        
        Args:
            ear_threshold (float): Eye Aspect Ratio threshold for determining eye openness
            max_num_faces (int): Maximum number of faces to detect simultaneously
        """
        self.detector = EyeOpennessDetector(ear_threshold=ear_threshold, max_num_faces=max_num_faces)
        self.frame_count = 0
        self.total_face_detections = 0
        self.awake_face_detections = 0
        self.closed_face_detections = 0
        
    def process_camera(self, camera_index=0):
        """
        Process real-time camera feed for eye openness detection
        
        Args:
            camera_index (int): Camera index (default: 0 for default camera)
        """
        print("Starting camera-based eye openness detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset statistics")
        print("  SPACE - Pause/Resume")
        
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect eye openness
                result = self.detector.detect_eye_openness(frame)
                
                # Update statistics
                self.frame_count += 1
                if result['faces_detected'] > 0:
                    for face_data in result['faces_data']:
                        self.total_face_detections += 1
                        if face_data['both_eyes_open']:
                            self.awake_face_detections += 1
                        else:
                            self.closed_face_detections += 1
                
                # Add statistics to the frame
                annotated_frame = self._add_statistics(result['annotated_image'], result['faces_detected'])
                
                # Display the frame
                cv2.imshow('Eye Openness Detection - Camera', annotated_frame)
                
                # Print real-time status
                if result['faces_detected'] > 0:
                    print(f"Frame {self.frame_count}: {result['faces_detected']} face(s) detected")
                    for face_data in result['faces_data']:
                        face_id = face_data['face_id']
                        status = "AWAKE" if face_data['both_eyes_open'] else "CLOSED/BLINKING"
                        print(f"  Face {face_id}: {status} | L_EAR: {face_data['left_ear']:.3f} | R_EAR: {face_data['right_ear']:.3f}")
                else:
                    print(f"Frame {self.frame_count}: No faces detected")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if 'result' in locals():
                    filename = f'camera_frame_{self.frame_count}.jpg'
                    cv2.imwrite(filename, result['annotated_image'])
                    print(f"Frame saved as '{filename}'")
            elif key == ord('r'):
                self._reset_statistics()
                print("Statistics reset")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        self._print_final_statistics()
    
    def process_video(self, video_path, output_path=None, save_output=True):
        """
        Process video file for eye openness detection
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path for output video (optional)
            save_output (bool): Whether to save the processed video
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file '{video_path}' not found")
            return
        
        print(f"Processing video: {video_path}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video writer if saving
        out = None
        if save_output:
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = f"{base_name}_eye_detection.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        print("\nProcessing frames...")
        print("Press 'q' to quit, SPACE to pause/resume")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect eye openness
                result = self.detector.detect_eye_openness(frame)
                
                # Update statistics
                self.frame_count += 1
                if result['faces_detected'] > 0:
                    for face_data in result['faces_data']:
                        self.total_face_detections += 1
                        if face_data['both_eyes_open']:
                            self.awake_face_detections += 1
                        else:
                            self.closed_face_detections += 1
                
                # Add statistics to the frame
                annotated_frame = self._add_statistics(result['annotated_image'], result['faces_detected'])
                
                # Save frame to output video
                if out is not None:
                    out.write(annotated_frame)
                
                # Display progress
                progress = (self.frame_count / total_frames) * 100
                awake_count = sum(1 for face in result['faces_data'] if face['both_eyes_open'])
                print(f"\rProgress: {progress:.1f}% | Frame {self.frame_count}/{total_frames} | "
                      f"Faces: {result['faces_detected']} | Awake: {awake_count}", end='')
                
                # Display the frame (resize for display if too large)
                display_frame = annotated_frame
                if width > 1024:
                    scale = 1024 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(annotated_frame, (new_width, new_height))
                
                cv2.imshow('Eye Openness Detection - Video', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")
        
        print("\nVideo processing complete!")
        
        # Clean up
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        self._print_final_statistics()
        
        if save_output and output_path:
            print(f"Processed video saved as: {output_path}")
    
    def process_image(self, image_path, output_path=None):
        """
        Process static image for eye openness detection
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path for output image (optional)
        """
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found")
            return
        
        print(f"Processing image: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from '{image_path}'")
            return
        
        # Detect eye openness
        result = self.detector.detect_eye_openness(image)
        
        # Print results
        print("\n=== Detection Results ===")
        if result['faces_detected'] > 0:
            print(f"Faces detected: {result['faces_detected']}")
            
            for face_data in result['faces_data']:
                face_id = face_data['face_id']
                print(f"\n--- Face {face_id} ---")
                print(f"Left eye open: {face_data['left_eye_open']} (EAR: {face_data['left_ear']:.3f})")
                print(f"Right eye open: {face_data['right_eye_open']} (EAR: {face_data['right_ear']:.3f})")
                print(f"Both eyes open: {face_data['both_eyes_open']}")
                
                # Determine status for this face
                if face_data['both_eyes_open']:
                    status = "AWAKE/ALERT"
                elif face_data['left_eye_open'] or face_data['right_eye_open']:
                    status = "BLINKING/WINKING"
                else:
                    status = "EYES CLOSED"
                print(f"Status: {status}")
            
            # Overall summary
            awake_faces = sum(1 for face in result['faces_data'] if face['both_eyes_open'])
            print(f"\n=== Summary ===")
            print(f"Total faces: {result['faces_detected']}")
            print(f"Awake faces: {awake_faces}")
            print(f"Closed/Blinking faces: {result['faces_detected'] - awake_faces}")
        else:
            print("No faces detected")
        
        # Save output image
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_eye_detection.jpg"
        
        cv2.imwrite(output_path, result['annotated_image'])
        print(f"Result saved as: {output_path}")
        
        # Display image
        print("\nPress any key to close the image window")
        cv2.imshow('Eye Openness Detection - Image', result['annotated_image'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def _add_statistics(self, image, faces_detected):
        """Add statistics overlay to the image"""
        if self.frame_count > 0 and self.total_face_detections > 0:
            awake_percentage = (self.awake_face_detections / self.total_face_detections) * 100
            stats_text = f"Frames: {self.frame_count} | Faces: {faces_detected} | Awake: {awake_percentage:.1f}%"
            cv2.putText(image, stats_text, (10, image.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return image
    
    def _reset_statistics(self):
        """Reset frame statistics"""
        self.frame_count = 0
        self.total_face_detections = 0
        self.awake_face_detections = 0
        self.closed_face_detections = 0
    
    def _print_final_statistics(self):
        """Print final processing statistics"""
        print(f"\n=== Final Statistics ===")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total face detections: {self.total_face_detections}")
        if self.total_face_detections > 0:
            awake_percentage = (self.awake_face_detections / self.total_face_detections) * 100
            print(f"Awake detections: {self.awake_face_detections} ({awake_percentage:.1f}%)")
            print(f"Closed/blinking detections: {self.closed_face_detections} ({100-awake_percentage:.1f}%)")
            print(f"Average faces per frame: {self.total_face_detections / self.frame_count:.1f}")


def main():
    """Main function to handle command line arguments and run the appropriate detection mode"""
    parser = argparse.ArgumentParser(description='Eye Openness Detection Script')
    parser.add_argument('--mode', choices=['camera', 'video', 'image'], required=True,
                       help='Detection mode: camera (webcam), video (file), or image (static)')
    parser.add_argument('--input', type=str,
                       help='Input file path (required for video and image modes)')
    parser.add_argument('--output', type=str,
                       help='Output file path (optional)')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='EAR threshold for eye openness detection (default: 0.1)')
    parser.add_argument('--camera-index', type=int, default=0,
                       help='Camera index for camera mode (default: 0)')
    parser.add_argument('--max-faces', type=int, default=5,
                       help='Maximum number of faces to detect simultaneously (default: 5)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output for video mode')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode in ['video', 'image'] and not args.input:
        print(f"Error: --input is required for {args.mode} mode")
        sys.exit(1)
    
    # Initialize the application
    app = EyeDetectionApp(ear_threshold=args.threshold, max_num_faces=args.max_faces)
    
    # Run the appropriate mode
    try:
        if args.mode == 'camera':
            app.process_camera(camera_index=args.camera_index)
        elif args.mode == 'video':
            app.process_video(args.input, args.output, save_output=not args.no_save)
        elif args.mode == 'image':
            app.process_image(args.input, args.output)
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
