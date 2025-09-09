import cv2
import time
import subprocess

class CameraHandler:
    def __init__(self, port=8080):
        self.stream_port = port
        self.stream_process = None
    
    def start_stream(self):
        """Start MJPG-Streamer"""
        cmd = f'export LD_LIBRARY_PATH=/usr/local/lib && mjpg_streamer -i "input_uvc.so -d /dev/video0 -r 640x480 -f 30" -o "output_http.so -p {self.stream_port}"'
        self.stream_process = subprocess.Popen(cmd, shell=True)
        print(f"Stream started on port {self.stream_port}")
    
    def stop_stream(self):
        """Stop MJPG-Streamer"""
        if self.stream_process:
            self.stream_process.terminate()
            print("Stream stopped")
    
    def capture_image(self, save_path):
        """Capture a single image from the camera"""
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(save_path, frame)
            print(f"Image saved to {save_path}")
        cap.release()
        return ret

if __name__ == "__main__":
    # Test the camera handler
    camera = CameraHandler()
    camera.start_stream()
    time.sleep(2)
    camera.capture_image("test_image.jpg")
    input("Press Enter to stop the stream...")
    camera.stop_stream()
