import cv2
import datetime
import numpy as np

camera = cv2.VideoCapture(0)
images = []
tic = datetime.datetime.now()
for i in range(10):
	return_value, image = camera.read()
	images.append(image)
	#cv2.imwrite('opencv'+str(i)+'.png', image)
del(camera)
toc = datetime.datetime.now()

dst = images[0]
for i in range(len(images)):
	if i == 0:
		pass
	else:
		alpha = 1.0/(i + 1)
		beta = 1.0 - alpha
		dst = cv2.addWeighted(images[i], alpha, dst, beta, 0.0)
 
# Save blended image
print(str(toc-tic))
cv2.imwrite('firstimg.png',images[0])
cv2.imwrite('averaged.png', dst)

greyImage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

alpha=3
beta=100

new_image=cv2.addWeighted(dst,alpha,np.zeros(dst.shape, dst.dtype),0,beta)
new_image2=cv2.addWeighted(greyImage,alpha,np.zeros(greyImage.shape, greyImage.dtype),0,beta)

cv2.imwrite('greyscale.png',greyImage)
cv2.imwrite('contboost.png', new_image)
cv2.imwrite('contboost2.png', new_image2)

normalized = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX)
normalized2 = cv2.normalize(new_image, None, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite('normalized.png',normalized)
cv2.imwrite('normalized2.png', normalized2)