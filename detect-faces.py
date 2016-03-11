#!/usr/bin/env python

# Face detection with OpenCV
#
# Usage: ./detect-faces.py la-garde.jpg /usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml
#
# Tutorial: https://realpython.com/blog/python/face-recognition-with-python/

import cv2
from datetime import datetime
import sys
import PIL.Image
import PIL.ExifTags

ROTATE_NONE = 0
ROTATE_CW   = 1
ROTATE_CCW  = 2

prefs = None

class Prefs:
    def __init__(self):
        self.draw_borders = True
        self.show_faces   = True
        self.border_ratio = 0.30
        self.square       = True
        #self.cascade      = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
        self.cascade      = '/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml'
        self.prefix       = 'face-'
        self.resize       = True
        self.min_size     = 48
        self.output_size  = 128

def read_image_exif(imagepath):
    # http://stackoverflow.com/questions/4764932/din-python-how-do-i-read-the-exif-data-for-an-image
    image = PIL.Image.open(imagepath)
    try:
        exif = {
            PIL.ExifTags.TAGS[k] : v
            for k, v in image._getexif().items()
            if k in PIL.ExifTags.TAGS
        }
        #print(exif)
        #print("  Date: ", exif['DateTimeOriginal'])
        #dt = datetime.strptime(exif['DateTimeOriginal'], "%Y:%m:%d %H:%M:%S")
        #print("  Date: %s") % (dt.strftime("%Y%m%d"))
        return exif
    except:
        return {}

def detect_faces(imagePath, faceCascade, index):
    print("Processing: " + imagePath)

    # Read image EXIF data to get image's original date
    exif = read_image_exif(imagePath)
    try:
        dt = datetime.strptime(exif['DateTimeOriginal'], "%Y:%m:%d %H:%M:%S")
        prefix = dt.strftime("%Y%m%d-") + prefs.prefix
    except:
        prefix = "00000000-" + prefs.prefix

    # Read the image
    image = cv2.imread(imagePath)

    # Determine if image needs to be rotated
    # EXIF orientations: http://jpegclub.org/exif_orientation.html
    # http://stackoverflow.com/questions/2259678/easiest-way-to-rotate-by-90-degrees-an-image-using-opencv
    need_rotate = ROTATE_NONE
    try:
        orientation = int(exif['Orientation'])
        if orientation == 6:
            need_rotate = ROTATE_CW
        elif orientation == 8:
            need_rotate = ROTATE_CCW
        elif orientation != 1:
            print("**** Orientation = %d ****") % (orientation)
        print("**** Orientation = %d ****") % (orientation)
    except:
        pass
    if need_rotate == ROTATE_CW:
        print ("  Rotate clockwise")
        image = cv2.transpose(image)
        image = cv2.flip(image, 1)
    elif need_rotate == ROTATE_CCW:
        print ("  Rotate counter-clockwise")
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)

    # Get image size
    image_height, image_width = image.shape[:2]

    # Show thumbnailed image
    if prefs.show_faces:
        tmp = cv2.resize(image, (image_width * 512 / image_height, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Image", tmp)
        cv2.moveWindow("Image", 500, 400)
        cv2.waitKey(200)

    # "Convert image to grayscale. Many operations in OpenCV are done in grayscale."
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        #minSize = (30, 30),
        minSize = (prefs.min_size, prefs.min_size),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    print("  Found %d faces:") % (len(faces))

    # Process the faces
    i = 0
    for (x, y, w, h) in faces:
        facepath = prefix + str(index + i) + ".png"
        print("  * x=%-4d y=%-4d w=%-4d h=%-4d --> %s") % (x, y, w, h, facepath)

        # Grow face limits to get more context (eg. 30% more than the longest dimension)
        border = int(max(h, w) * prefs.border_ratio / 2.0)
        if not prefs.square:
            x = max(x - border, 0)
            y = max(y - border, 0)
            w = min(w + border * 2, image_width)
            h = min(h + border * 2, image_height)
        else:
            if h > w:
                x = int(x - ((h - w) / 2))
            elif h < w:
                y = int(y - ((w - h) / 2))
            x = max(x - border, 0)
            y = max(y - border, 0)
            w = min(max(h, w) + border * 2, image_width)
            h = min(max(h, h) + border * 2, image_height)

        # Crop face
        # http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
        face = image[y:y+h, x:x+w]

        # Resize face
        if prefs.resize:
            face = cv2.resize(face,
                              (prefs.output_size, prefs.output_size),
                              #interpolation=cv2.INTER_CUBIC)
                              interpolation=cv2.INTER_LANCZOS4)

        # Save face
        cv2.imwrite(facepath, face)

        if prefs.show_faces:
            cv2.imshow("Face", face)
            cv2.moveWindow("Face", 500, 400)
            cv2.waitKey(500)

        i += 1

    # Draw a rectangle around the faces in original image
    if prefs.draw_borders:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
        tmp = cv2.resize(image, (image_width * 512 / image_height, 512), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Image", tmp)
        cv2.moveWindow("Image", 500, 400)
        cv2.waitKey(500)

    # Destroy window before processing next image
    if prefs.show_faces or prefs.draw_borders:
        cv2.destroyWindow("Image")
    if prefs.show_faces:
        cv2.destroyWindow("Face")

    # Return number of detected faces
    return i

# Image: OpenCV to PIL
# http://stackoverflow.com/questions/13576161/convert-opencv-image-into-pil-image-in-python-for-use-with-zbar-library
def cv2_to_pil():
    cv2_im = cv2.imread(imagePath)
    # Do OpenCV operations
    cv2_im.cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    pil_im.show()

# Image: PIL to OpenCV
# http://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
def pil_to_cv2():
    pil_im = PIL.Image.open(imagePath)
    # Do PIL operations
    cv2_im = cv2.cvtColor(numpy.array(pil_im), cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", cv2_im)


def main():
    global prefs
    prefs = Prefs()

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(prefs.cascade)

    # Detect faces in each image passed in argument
    index = 0
    for imagePath in sys.argv[1:]:
        faces = detect_faces(imagePath, faceCascade, index)
        index += faces

if __name__ == "__main__":
    main()


