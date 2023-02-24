# Import required packages 
import numpy as np
import cv2 
import pytesseract 
from PIL import Image


FILENAME = "samples/image.jpg"

def filterOutSaltPepperNoise(edgeImg):
	# Get rid of salt & pepper noise.
	count = 0
	lastMedian = edgeImg
	median = cv2.medianBlur(edgeImg, 3)
	while not np.array_equal(lastMedian, median):
		# get those pixels that gets zeroed out
		zeroed = np.invert(np.logical_and(median, edgeImg))
		edgeImg[zeroed] = 0

		count = count + 1
		if count > 70:
			break
		lastMedian = median
		median = cv2.medianBlur(edgeImg, 3)

def findSignificantContour(edgeImg):
	contours, hierarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# Find level 1 contours
	level1Meta = []
	for contourIndex, tupl in enumerate(hierarchy[0]):
		# Each array is in format (Next, Prev, First child, Parent)
		# Filter the ones without parent
		if tupl[3] == -1:
			tupl = np.insert(tupl.copy(), 0, [contourIndex])
			level1Meta.append(tupl)
		# From among them, find the contours with large surface area.
	contoursWithArea = []
	for tupl in level1Meta:
		contourIndex = tupl[0]
		contour = contours[contourIndex]
		area = cv2.contourArea(contour)
		contoursWithArea.append([contour, area, contourIndex])
		
	contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
	largestContour = contoursWithArea[0][0]
	return largestContour

def remove_background():

	src = cv2.imread(FILENAME, 1)
	blurred = cv2.GaussianBlur(src, (5, 5), 0)
	blurred_float = blurred.astype(np.float32) / 255.0
	edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("model.yml")
	edges = edgeDetector.detectEdges(blurred_float) * 255.0
	cv2.imwrite('edge-raw.jpg', edges)
	edges_8u = np.asarray(edges, np.uint8)
	filterOutSaltPepperNoise(edges_8u)
	cv2.imwrite('edge.jpg', edges_8u)
	contour = findSignificantContour(edges_8u)
	# Draw the contour on the original image
	contourImg = np.copy(src)
	cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
	cv2.imwrite('contour.jpg', contourImg)
	mask = np.zeros_like(edges_8u)
	cv2.fillPoly(mask, [contour], 255)

	# calculate sure foreground area by dilating the mask
	mapFg = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=10)

	# mark inital mask as "probably background"
	# and mapFg as sure foreground
	trimap = np.copy(mask)
	trimap[mask == 0] = cv2.GC_BGD
	trimap[mask == 255] = cv2.GC_PR_BGD
	trimap[mapFg == 255] = cv2.GC_FGD

	# visualize trimap
	trimap_print = np.copy(trimap)
	trimap_print[trimap_print == cv2.GC_PR_BGD] = 128
	trimap_print[trimap_print == cv2.GC_FGD] = 255
	cv2.imwrite('trimap.png', trimap_print)
	# run grabcut
	bgdModel = np.zeros((1, 65), np.float64)
	fgdModel = np.zeros((1, 65), np.float64)
	rect = (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)
	cv2.grabCut(src, trimap, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

	# create mask again
	mask2 = np.where(
		(trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD),
		255,
		0
	).astype('uint8')
	cv2.imwrite('mask2.jpg', mask2)
	contour2 = findSignificantContour(mask2)
	mask3 = np.zeros_like(mask2)
	cv2.fillPoly(mask3, [contour2], 255)
	# blended alpha cut-out
	mask3 = np.repeat(mask3[:, :, np.newaxis], 3, axis=2)
	mask4 = cv2.GaussianBlur(mask3, (3, 3), 0)
	alpha = mask4.astype(float) * 1.1  # making blend stronger
	alpha[mask3 > 0] = 255.0
	alpha[alpha > 255] = 255.0

	foreground = np.copy(src).astype(float)
	foreground[mask4 == 0] = 0
	background = np.ones_like(foreground, dtype=float) * 255.0

	cv2.imwrite('foreground.png', foreground)
	cv2.imwrite('background.png', background)
	cv2.imwrite('alpha.png', alpha)

	# Normalize the alpha mask to keep intensity between 0 and 1
	alpha = alpha / 255.0
	# Multiply the foreground with the alpha matte
	foreground = cv2.multiply(alpha, foreground)
	# Multiply the background with ( 1 - alpha )
	background = cv2.multiply(1.0 - alpha, background)
	# Add the masked foreground and background.
	cutout = cv2.add(foreground, background)

	cv2.imwrite('cutout.jpg', cutout)

def showWindow(img):
	cv2.namedWindow('image',cv2.WINDOW_NORMAL)
	cv2.resizeWindow('image', 600,800)
	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def main():
	# Mention the installed location of Tesseract-OCR in your system 
	pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'


	# Preprocessing the image starts 
	remove_background() 
	img = cv2.imread('cutout.jpg', 1)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



	# Performing OTSU threshold 
	ret, thresh1 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV) 

	# Specify structure shape and kernel size. 
	# Kernel size increases or decreases the area 
	# of the rectangle to be detected. 
	# A smaller value like (10, 10) will detect 
	# each word instead of a sentence. 
	rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 

	# Appplying dilation on the threshold image 
	dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 

	# Finding contours 
	contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
													cv2.CHAIN_APPROX_NONE) 

	# Creating a copy of image 
	im2 = img.copy() 

	# A text file is created and flushed 
	file = open("recognized.txt", "w+") 
	file.write("") 
	file.close() 

	# Looping through the identified contours 
	# Then rectangular part is cropped and passed on 
	# to pytesseract for extracting text from it 
	# Extracted text is then written into the text file 
	for cnt in contours: 
		x, y, w, h = cv2.boundingRect(cnt) 
		
		# Drawing a rectangle on copied image 
		rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
		# cv2.namedWindow('contour',cv2.WINDOW_NORMAL)
		# cv2.resizeWindow('contour', 600, 800)
		# cv2.imshow('contour', rect)
		# cv2.waitKey(0)  
		# cv2.destroyAllWindows()  
		# Cropping the text block for giving input to OCR 
		cropped = im2[y:y + h, x:x + w] 
		
		# Open the file in append mode 
		file = open("recognized.txt", "a") 
		
		# Apply OCR on the cropped image 
		text = pytesseract.image_to_string(cropped) 

		# Appending the text into file 
		file.write(text) 
		file.write("\n") 
		
		# Close the file 
		file.close 
	# print(pytesseract.image_to_string(Image.open('cutout.jpg'),lang="eng"))
# preprocessing
# gray scale
def gray(img):
	
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(r"./preprocess/img_gray.png",img)
	return img

# blur
def blur(img) :
	img_blur = cv2.GaussianBlur(img,(5,5),0)
	cv2.imwrite(r"./preprocess/img_blur.png",img)    
	return img_blur

# threshold
def threshold(img):
	#pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
	img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]    
	cv2.imwrite(r"./preprocess/img_threshold.png",img)
	return img

def alternative():
	im = cv2.imread(FILENAME, 1)
	# Finding contours 
	im_gray = gray(im)
	im_blur = blur(im_gray)
	im_thresh = threshold(im_blur)
	contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
	# text detection
	i = 0
	for cnt in contours:
		i += 1
		x, y, w, h = cv2.boundingRect(cnt) 

		# Drawing a rectangle on copied image 
		rect = cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2) 
		
		# showWindow(rect)

		# Cropping the text block for giving input to OCR 
		cropped = im[y:y + h, x:x + w] 

		# Apply OCR on the cropped image 
		config = ('-l eng --oem 1 --psm 3')
		text = pytesseract.image_to_string(cropped, config=config) 
	print(i)
alternative()
