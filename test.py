from ultralytics import YOLO

model = YOLO("runs/classify/train-2/weights/best.pt")

result = model("test.jpg")

pred = result[0]

print("결과:", pred.names[pred.probs.top1])
print("확률:", float(pred.probs.top1conf))