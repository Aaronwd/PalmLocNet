import numpy as np
import cv2

savepath = '/home/aaron/桌面/video-pic-test/xiaoxxx/xiaoxxx' #记得检查
cap = cv2.VideoCapture(0)

#保存成视频的时候无法播放，需要进一步检查
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output.mp4',fourcc,20.0, (64,64))

f_index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        #掉转视频的方向
        frame = cv2.flip(frame,1)
#        out.write(frame)
        cv2.imshow('frame',frame)
        print(frame.shape)
        #resize每一帧图片并且保存
        frame_resize = cv2.resize(frame, (480,480),interpolation=cv2.INTER_AREA)
        cv2.imwrite(savepath+str(f_index)+'.jpg', frame_resize)
        f_index += 1
        if f_index == 500 or (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    else:
        break

cap.release()
#out.release()
cv2.destroyAllWindows()

