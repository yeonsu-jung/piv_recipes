# %%
import cv2
# %%
vidcap = cv2.VideoCapture('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-19/RoughSphere_3cmAway_motor10_timing400.mp4')

success,image = vidcap.read()
success
# %%

grayFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# %%
import os
os.mkdir('C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-19/RoughSphere_3cmAway_motor10_timing400')
# %%
count = 0

while success:
  cv2.imwrite("C:/Users/yj/Dropbox (Harvard University)/Riblet/data/piv-data/2021-04-19/RoughSphere_3cmAway_motor10_timing400/frame_%06d.tiff" %count, grayFrame)     # save frame as JPEG file      
  success,image = vidcap.read()
  grayFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  print('Read a new frame: ', success)
  count += 1
# %%
