import cv2
import numpy as np

def whiten_teeth(yellow_teeth_photo):
	"""
	receives a photo with yellow teeth and returns a photo with whitened teeth
	"""
	whitened_teeth_photo = yellow_teeth_photo.copy()

	mouth_cascade = cv2.CascadeClassifier("Mouth.xml")
	eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

	faces = face_cascade.detectMultiScale(yellow_teeth_photo, 1.3, 5)

	# if there is no face, then the image may be a zoomed in image of teeth,
	# so we need to check for that possibility as well
	if len(faces) != 0:

		# if there are faces, go through each of the faces and find the mouth
		for (x, y, w, h) in faces:
			cv2.rectangle(yellow_teeth_photo, (x, y), (x + w, y + h), (0, 255, 0), 2)
			face_section = yellow_teeth_photo[y:y + h, x: x + w]
			FaceFileName = "unknowfaces/face_" + str(y) + ".jpg"
			cv2.imwrite(FaceFileName, face_section)

			mouths = mouth_cascade.detectMultiScale(face_section, 1.3, 5)
			for (mx, my, mw, mh) in mouths:

				# sometimes eyes get picked up as a potential mouth,
				# so if the y coordinate is above halfway down the face,
				# it is most likely an eye, so we skip it
				if my < face_section.shape[1] / 2:
					continue

				#cv2.rectangle(face_section, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)  #uncomment to draw rectangle

				# crop the face to the mouth, then call the core processing algorithm,
				# then place the new mouth into the face
				mouth_section = face_section[my:my + mh, mx:mx + mw]
				mouth_sec = face_section[my:my + (mh+5), mx:mx + (mw+5)] # +5 arbitrarily to save complete mouth crop
				MouthFileName = "unknowmouths/mouth_" + str(my) + ".jpg"
				cv2.imwrite(MouthFileName, mouth_sec)				
				
				whitened_mouth = whiten_teeth_core(mouth_section)
				face_section[my:my + mh, mx:mx + mw] = whitened_mouth

			yellow_teeth_photo[y:y + h, x: x + w] = face_section

	else:
		# the image is most likely a zoomed in face, where we cannot see the entire face
		# so we search for mouths independently of any found faces and choose the largetst potential mouth
		mouths = mouth_cascade.detectMultiScale(yellow_teeth_photo, 1.3, 5)
		mouth = get_largest_mouth_region(mouths)

		mx, my, mw, mh = mouth
		#cv2.rectangle(yellow_teeth_photo, (mx, my), (mx + mw, my + mh), (255, 0, 0), 2)

		mouth_section = yellow_teeth_photo[my:my + mh, mx:mx + mw]
		mouth_sec = yellow_teeth_photo[my:my + (mh+5), mx:mx + (mw+5)]  
		MouthFileName = "unknowmouths/mouth_" + str(my) + ".jpg"
		cv2.imwrite(MouthFileName, mouth_sec)
		whitened_mouth = whiten_teeth_core(mouth_section)
		whitened_teeth_photo[my:my + mh, mx:mx + mw] = whitened_mouth

	return whitened_teeth_photo

def whiten_teeth_core(mouth_section):
	"""
	Core algorithm for detecting teeth within the mouth and whitening them
	"""
	blue, green, red = cv2.split(mouth_section)
	red_green_sum = cv2.add(red, green)

	# create our mask where red and green are strong and blue is weak, this is a yellow area
	mask = (red > 150) & (green > 150) & (blue < 230)
	mask = mask.astype(np.uint8) * 255

	# find our mouth contour
	_, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	mouth_contour = get_largest_contour(contours)

	mask = resetMask(mask)

	cv2.drawContours(mask, [mouth_contour], 0, 255, -1)

	# expand the mouth area to cover uncaught edges
	kernel = np.ones((3, 3), np.uint8)
	mask = cv2.dilate(mask, kernel, iterations=2)

	# adjust the 
	whiter_color_replacement = configure_white(red_green_sum, mask)

	# reconvert the mask to the BGR color space and invert it to fill the original red eye hole
	mask_inverse = ~cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
	whitened_mouth = cv2.bitwise_and(mask_inverse, mouth_section) + whiter_color_replacement

	return whitened_mouth

def configure_white(red_green_sum, mask):
    """
    Helper function for adjusting the blue-green color mean for blending the old red eye
    """
    whiter_color = red_green_sum
    for color_set in whiter_color:
    	for i in range(0, len(color_set)):
    		color = color_set[i]
    		color_set[i] = color * .92
    whiter_color = cv2.bitwise_and(whiter_color, mask)
    whiter_color = cv2.cvtColor(whiter_color, cv2.COLOR_GRAY2BGR)
    return whiter_color


def resetMask(current_mask):
    """
    Resets the mask to its default value
    """
    return current_mask * 0


def get_largest_contour(contours):
    """
    In the given images, there may be various sections picked up by the detection algorithm,
    this chooses the largest section to correct, as it is most likely the actual red eye
    """
    largest_contour_area = 0
    largest_contour = None

    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area > largest_contour_area:
            largest_contour_area = contour_area
            largest_contour = contour

    return largest_contour

def get_largest_mouth_region(mouths):
	"""
	Gets the largest mouth region on a zoomed in picture of a face when there
	may be multiple possible mouths selected by the haarcascade
	"""
	largest_region = 0
	largest_mouth = None

	for mouth in mouths:
		mx, my, mw, mh = mouth
		area = (mx + mw) * (my + mh)
		if area > largest_region:
			largest_region = area
			largest_mouth = mouth

	return largest_mouth


