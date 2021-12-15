import cv2
import numpy as np

class Preprocess:

    def histogram_equlization_rgb(self, img):
        # Simple preprocessing using histogram equalization 
        # https://en.wikipedia.org/wiki/Histogram_equalization
        print(img.shape)
        intensity_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        intensity_img[:, :, 0] = cv2.equalizeHist(intensity_img[:, :, 0])
        img = cv2.cvtColor(intensity_img, cv2.COLOR_YCrCb2BGR)

        # For Grayscale this would be enough:
        # img = cv2.equalizeHist(img)

        return img

    # Add your own preprocessing techniques here.
    def brightness_correction(self, img):
        cols, rows, _ = img.shape
        brightness = np.sum(img) / (255 * cols * rows)
        minimum_brightness = 0.66

        ratio = brightness / minimum_brightness
        if ratio >= 1:
            print("Image already bright enough")
            return img

        # Otherwise, adjust brightness to get the target brightness
        return cv2.convertScaleAbs(img, alpha=1 / ratio, beta=0)

    def sharpen_image(self, img):
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])

        image_sharp = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)

        return image_sharp
