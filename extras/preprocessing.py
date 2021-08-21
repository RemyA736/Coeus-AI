import cv2
import pytesseract
import urllib
import numpy as np
import re

# BGR a GrayScale
def rgb_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Se cambia el tamaño de la imagen (downscale)
def downscale_img(img, factor):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

# Se remueven sombras
def shadow_removal(img):
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)
    img = cv2.merge(result_planes)

    return img

# Eliminación de ruido
def noise_removal(img):
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1) # increases the white region in the image
    img = cv2.erode(img, kernel, iterations=1) # erodes away the boundaries of foreground object

    return img

# Inversión de los valores en la imagen
def invert(img):
    return cv2.bitwise_not(img)

# Binarización de Otsu
def otsu_binarization(cv2_img):
    return cv2.threshold(cv2_img, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Binarización adaptativa
def adaptive_binarization(cv2_img):
    return cv2.adaptiveThreshold(cv2_img, 255,
                             cv2.ADAPTIVE_THRESH_MEAN_C,
                             cv2.THRESH_BINARY_INV,21,10)

def deskew_tesseract(image):
    #  TAKEN FROM: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"

    rot_data = pytesseract.image_to_osd(image, lang='spa', config='--psm 0 --dpi 300');
    rot = re.search('(?<=Rotate: )\d+', rot_data).group(0)

    angle = float(rot)
    if angle == 0:
        return image  # REVISAR SI FUNCIONA
    if angle == 90:
        return cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 360:
		return cv2.rotate(image, cv2.cv2.ROTATE_180) # REVISAR SI FUNCIONA
        

def deskew_cv2(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)
    angle = minAreaRect[-1]
    if angle > 0:
        angle = 360 - angle
    else:
        pass
    if angle == 90:
        return cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return image

def remove_borders(image):
	mask = np.zeros(image.shape, dtype=np.uint8)

	cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,
							cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]

	cv2.fillPoly(mask, cnts, [255,255,255])
	mask = 255 - mask

	return cv2.bitwise_or(image, mask)

def crop_to_text(image):
	img = image

	d = pytesseract.image_to_data(img,
								  output_type=pytesseract.Output.DICT,
								  config='-psm 9')
	n_boxes = len(d['level'])
	for i in range(n_boxes):
		(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cropped_image = img[y:y+h, x:x+w]

def preprocess(image, binarization='adaptive', ):
	img = rgb_to_gray(image)
	img = downscale_img(img, 0.7)
	img = shadow_removal(img)
	
	# Tipo de binarización
	if binarization = 'otsu':
		T, img_otsu = otsu_binarization(img)
		img = invert(img_otsu)
	else:
		img_binaria = adaptive_binarization(img)
		img = invert(img_binaria)
	
	