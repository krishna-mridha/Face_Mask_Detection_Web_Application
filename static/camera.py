import cv2
class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret, frame = self.video.read()
        flag = 0
        frame = imutils.resize(frame, width=400)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            
            label = "Mask" if mask > withoutMask else "No Mask"
            if label == "Mask":
                color = (0, 255, 0)
                winsound.PlaySound(None, winsound.SND_ASYNC)
                flag = 0
            else:
                color = (0, 0, 255)
                if flag==0:
                    print('Wear your Mask!')
                    winsound.PlaySound("alarm.wav", winsound.SND_ASYNC | winsound.SND_ALIAS )
                    flag=1

        ret, jpg = cv2.imencode('.jpg',frame)
        return jpg.tobytes()
