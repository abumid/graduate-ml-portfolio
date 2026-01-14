#!/usr/bin/env python3
"""
Quick Test Script for Eye Detection System
This script tests the eye detection functionality without requiring user interaction.
"""

import cv2
import os
from eyelibuz.eye_openness import EyeOpennessDetector

def test_eye_detection():
    """Test the eye detection system with available images"""
    
    print("=== Eye Detection System Test ===\n")
    
    # Initialize detector
    detector = EyeOpennessDetector(ear_threshold=0.1)
    
    # Find available test images
    image_extensions = ['.jpg', '.jpeg', '.png']
    test_images = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            test_images.append(file)
    
    if not test_images:
        print("No test images found in current directory.")
        return
    
    print(f"Found {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img}")
    
    print("\nTesting each image...\n")
    
    results_summary = []
    
    for i, image_path in enumerate(test_images, 1):
        print(f"[{i}/{len(test_images)}] Processing: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  ❌ Could not load image")
            continue
        
        # Detect eye openness
        result = detector.detect_eye_openness(image)
        
        # Display results
        if result['face_detected']:
            left_status = "OPEN" if result['left_eye_open'] else "CLOSED"
            right_status = "OPEN" if result['right_eye_open'] else "CLOSED"
            overall_status = "AWAKE" if result['both_eyes_open'] else "DROWSY/BLINKING"
            
            print(f"  ✅ Face detected")
            print(f"  👁️  Left eye: {left_status} (EAR: {result['left_ear']:.3f})")
            print(f"  👁️  Right eye: {right_status} (EAR: {result['right_ear']:.3f})")
            print(f"  📊 Overall: {overall_status}")
            
            results_summary.append({
                'image': os.path.basename(image_path),
                'face_detected': True,
                'both_eyes_open': result['both_eyes_open'],
                'left_ear': result['left_ear'],
                'right_ear': result['right_ear']
            })
        else:
            print(f"  ❌ No face detected")
            results_summary.append({
                'image': os.path.basename(image_path),
                'face_detected': False,
                'both_eyes_open': False,
                'left_ear': 0.0,
                'right_ear': 0.0
            })
        
        print()
    
    # Print summary
    print("=== Test Summary ===")
    faces_detected = sum(1 for r in results_summary if r['face_detected'])
    eyes_open = sum(1 for r in results_summary if r['both_eyes_open'])
    
    print(f"Total images tested: {len(results_summary)}")
    print(f"Faces detected: {faces_detected}/{len(results_summary)} ({faces_detected/len(results_summary)*100:.1f}%)")
    
    if faces_detected > 0:
        print(f"Eyes open: {eyes_open}/{faces_detected} ({eyes_open/faces_detected*100:.1f}%)")
        
        # Average EAR values
        avg_left_ear = sum(r['left_ear'] for r in results_summary if r['face_detected']) / faces_detected
        avg_right_ear = sum(r['right_ear'] for r in results_summary if r['face_detected']) / faces_detected
        print(f"Average EAR - Left: {avg_left_ear:.3f}, Right: {avg_right_ear:.3f}")
    
    print(f"\nTest completed! Check generated '*_eye_detection.jpg' files for visual results.")

def test_detector_initialization():
    """Test if the detector can be initialized properly"""
    print("Testing detector initialization...")
    
    try:
        detector = EyeOpennessDetector()
        print("✅ EyeOpennessDetector initialized successfully")
        
        # Test with different thresholds
        detector_sensitive = EyeOpennessDetector(ear_threshold=0.15)
        detector_conservative = EyeOpennessDetector(ear_threshold=0.3)
        print("✅ Different threshold configurations work")
        
        return True
    except Exception as e:
        print(f"❌ Detector initialization failed: {e}")
        return False

def main():
    """Main test function"""
    print("Starting Eye Detection System Tests...\n")
    
    # Test 1: Detector initialization
    if not test_detector_initialization():
        print("Basic initialization test failed. Cannot proceed.")
        return
    
    print()
    
    # Test 2: Image processing
    test_eye_detection()
    
    print("\n=== All Tests Completed ===")
    print("The eye detection system is ready to use!")
    print("\nNext steps:")
    print("1. Run 'python eye_detection_demo.py' for interactive demo")
    print("2. Use 'python eye_detection_script.py --mode camera' for live detection")
    print("3. Check the EYE_DETECTION_README.md for detailed usage instructions")

if __name__ == "__main__":
    main()
