import cv2, glob
files = sorted(glob.glob("demo_photos/*.jpg"))
h, w = cv2.imread(files[0]).shape[:2]
out = cv2.VideoWriter("demo.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 1, (w, h))
for f in files:
    img = cv2.resize(cv2.imread(f), (w, h))
    for _ in range(3):
        out.write(img)
out.release()
print("demo.mp4 ready")