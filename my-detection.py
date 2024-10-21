from jetson.inference import detectNet
from jetson.utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

net = detectNet("ssd-mobilenet-v2", threshold=0.5)
img_path = "/home/nvidia/a3/Image_8.jpg"
img = jetson.utils.loadImage(img_path)
display = videoOutput("display://0") 
if img is None:  
    print("nothing to detect")
    KeyboardInterrupt

while True:
    detections = net.Detect(img)
    display.Render(img)
    for detection in detections:
        print(detection)

