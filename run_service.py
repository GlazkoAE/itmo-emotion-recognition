import cv2
from face.inference import Model as FaceModel
#from pose.model.inferense import ArousalModel
#from service.drawer import Drawer
#from service.image_loader import ImageTransformer
#from voice.run_model import ModelResnetForAudio
from moviepy.editor import VideoFileClip

from concurrent import futures
import numpy as np
import grpc
import time
import sys
import pickle

from face.faceProcessing import EmotionDetector

#sys.path.append("/usr/app/grpc_compiled")
from service import service_pb2 as spb2
from service import service_pb2_grpc as spb2grpc


class EService(spb2grpc.RequestHandlerServicer):

    def GetEncode(self, request, context):
        #print("Received job !")
        frame = pickle.loads(request.image)
        #emoDetector.set_wh(request.width, request.height)
        
        emo_predict, box = face_model.predict(frame)
        emo_predict = (
            "Face was not founded"
            if emo_predict is None
            else emo_predict
        )
        
        if box is not None:
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

        
        #print(emo_predict)        #image_transformed = image_to_negative(image)
        return spb2.processedAndPrediction(image=pickle.dumps(frame), prediction=emo_predict)

if __name__=="__main__":

    #image_loader = ImageTransformer()
    face_model = FaceModel()
    #voice_model = ModelResnetForAudio()
    #pose_model = ArousalModel(saved_model="./pose/best-lstm.hdf5")
    #drawer = Drawer()
    #emoDetector = EmotionDetector(300, 300)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    spb2grpc.add_RequestHandlerServicer_to_server(EService(),server)
    server.add_insecure_port('[::]:13000')
    server.start()
    print("Server started. Awaiting jobs...")
    try:
        while True: # since server.start() will not block, a sleep-loop is added to keep alive
            time.sleep(60*60*24)
            
    except KeyboardInterrupt:
        server.stop(0)

