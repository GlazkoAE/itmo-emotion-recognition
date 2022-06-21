import cv2
import torch 
from torchvision import transforms
import numpy as np
if __name__ == "__main__":
    draw_bbox = True
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
    class_names = {0:'Surprise', 
                   1:'Fear', 
                   2:'Disgust', 
                   3:'Happiness', 
                   4:'Sadness', 
                   5:'Anger', 
                   6:'Neutral'}    
    
    cap = cv2.VideoCapture(0)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    face_bboxer = cv2.dnn.readNetFromCaffe('../models/face/deploy.prototxt',
                    '../models/face/res10_300x300_ssd_iter_140000.caffemodel')
    emotion_classifier = torch.jit.load('../models/face/ResnetRUL_cuda.pth')
    
    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get face_crop
        face_bboxer.setInput(cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                                (300, 300), (104.0, 117.0, 123.0)))
        bboxes = face_bboxer.forward()
         
        for i in range(bboxes.shape[2]):
            confidence = bboxes[0, 0, i, 2]
            if confidence > 0.5:
                box = bboxes[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int") 
                if (0<y<y1<h) and (0<x<x1<w):
                    crop = image[y:y1, x:x1]
                    topred = transform(crop).unsqueeze(0).to('cuda:0')
                    pred = emotion_classifier(topred).argmax(axis=-1)
                    print(class_names[pred.item()])
                else:
                    crop = None
                if draw_bbox:
                    cv2.rectangle(image, (x, y), (x1, y1), (0, 0, 255), 2)
                    
        # Flip the image horizontally for a selfie-view display
        cv2.imshow('Faceial emotion detection demo', cv2.flip(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 1))

        # Flip the image horizontally for a selfie-view display
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()