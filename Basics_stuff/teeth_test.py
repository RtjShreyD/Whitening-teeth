import argparse
import teeth_whitener
import cv2
import numpy as np

if __name__=="__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="path to the input image")
	args = vars(ap.parse_args())
		
	yellow_teeth_photo = cv2.imread(args["image"])

	fixed_image = teeth_whitener.whiten_teeth(yellow_teeth_photo)

	result = np.hstack((yellow_teeth_photo, fixed_image))

	cv2.imshow("Teeth Whitening", result)
	cv2.waitKey()
	cv2.destroyAllWindows()
