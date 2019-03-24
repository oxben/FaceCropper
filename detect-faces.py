#!/usr/bin/env python3

'''
Face detection with OpenCV

Usage: ./detect-faces.py image_file1 image_file2 ...

Author: Oxben <oxben@free.fr>

Based on OpenCV tutorial: https://realpython.com/blog/python/face-recognition-with-python/
'''

import cv2
from datetime import datetime
import logging
import os.path
import sys
import PIL.Image
import PIL.ExifTags

import levels

ROTATE_NONE = 0
ROTATE_CW   = 1
ROTATE_CCW  = 2

prefs = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------
class Stats:
    '''Statistics class'''

    def __init__(self):
        self.nfiles = 0
        self.nfaces = 0
        self.skip_image_too_small = 0
        self.skip_face_too_small  = 0

#-------------------------------------------------------------------------------
class Prefs:
    '''Preferences class'''

    def __init__(self):
        self.draw_borders   = True
        self.show_faces     = False
        self.border_ratio   = 0.40
        self.square         = True
        #self.cascade       = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
        self.cascade        = '/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml'
        #self.cascade        = '/usr/share/opencv/haarcascades/haarcascade_profileface.xml'
        self.prefix         = 'face'
        self.resize         = True
        self.min_image_size = 500
        self.min_face_size  = 64
        self.output_size    = 256
        self.skip_exist     = True
        self.auto_adjust    = True

#-------------------------------------------------------------------------------
def read_image_exif(imagepath):
    '''
    Read EXIF data from an image
    Based on: http://stackoverflow.com/questions/4764932/din-python-how-do-i-read-the-exif-data-for-an-image
    '''
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
        #print("  Date: %s" % (dt.strftime("%Y%m%d")))
        return exif
    except:
        return {}

#-------------------------------------------------------------------------------
def detect_faces(imagePath, faceCascade):
    ''' Detect faces in an image'''
    print("Processing: " + imagePath)

    # Read image EXIF data to get image's original date
    # If there's no EXIF, use file modification time
    exif = read_image_exif(imagePath)
    try:
        dt = datetime.strptime(exif['DateTimeOriginal'], "%Y:%m:%d %H:%M:%S")
    except:
        dt = datetime.fromtimestamp(os.path.getmtime(imagePath))
    prefix = prefs.prefix + dt.strftime("-%Y%m%d-%H%M%S")

    # Check if image has already been processed (ie. a face already exists for it)
    if prefs.skip_exist and os.path.isfile(prefix + '-0.jpg'):
        print("  Skip. Image has already been processed before (%s)" % (prefix + '-0.jpg'))
        return 0

    # Read the image
    image = cv2.imread(imagePath)
    if image is None:
        print("  Skip. Cannot load image.")
        return 0

     # Get image size
    image_height, image_width = image.shape[:2]

    # Skip small image
    if image_height < prefs.min_image_size or image_width < prefs.min_image_size:
        print("  Skip. Image too small (%dx%d)." % (image_height, image_width))
        stats.skip_image_too_small += 1
        return 0

    if False:
        # OpenCV 3.x seems to handle orientation by itself
        #
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

    # Update image size
    image_height, image_width = image.shape[:2]

    # Show thumbnailed image
    if prefs.show_faces:
        tmp = cv2.resize(image, (int(image_width * 512 / image_height), 512), interpolation=cv2.INTER_NEAREST)
        cv2.moveWindow("Image", 500, 400)
        cv2.imshow("Image", tmp)
        cv2.waitKey(1000)

    # "Convert image to grayscale. Many operations in OpenCV are done in grayscale."
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 3,
        minSize = (prefs.min_face_size, prefs.min_face_size),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    print("  Found %d faces:" % (len(faces)))

    # Process the faces
    i = 0
    for (x, y, w, h) in faces:
        stats.nfaces += 1

        facepath = prefix + '-' + str(i) + ".jpg"
        print("  * x=%-4d y=%-4d w=%-4d h=%-4d --> %s" % (x, y, w, h, facepath))

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

        # Skip face smaller than output size
        if w < prefs.output_size or h < prefs.output_size:
            print("    Skip. Face too small")
            stats.skip_face_too_small += 1
            continue

        # Crop face
        # http://stackoverflow.com/questions/15589517/how-to-crop-an-image-in-opencv-using-python
        face = image[y:y+h, x:x+w]

        # Resize face
        if prefs.resize:
            face = cv2.resize(face,
                              (prefs.output_size, prefs.output_size),
                              interpolation=cv2.INTER_CUBIC)
                              #interpolation=cv2.INTER_LANCZOS4)

        # Save face
        cv2.imwrite(facepath, face, [cv2.IMWRITE_JPEG_QUALITY, 93])

        if prefs.auto_adjust:
            im1 = PIL.Image.open(facepath)
            im2 = levels.levels(im1)
            im2.save(facepath, "JPEG", quality=93)

        if prefs.show_faces:
            cv2.imshow("Face", face)
            cv2.moveWindow("Face", 500, 400)
            cv2.waitKey(500)
        i += 1

    # Draw a rectangle around the faces in original image
    if prefs.draw_borders:
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 4)
        tmp = cv2.resize(image, (int(image_width * 512 / image_height), 512), interpolation=cv2.INTER_NEAREST)
        cv2.moveWindow("Image", 500, 400)
        cv2.imshow("Image", tmp)
        cv2.waitKey(1000)

    # Destroy window before processing next image
    if prefs.show_faces or prefs.draw_borders:
        cv2.destroyWindow("Image")
    if prefs.show_faces:
        cv2.destroyWindow("Face")

    # Return number of detected faces
    return i

#-------------------------------------------------------------------------------
def cv2_to_pil():
    '''
    Convert OpenCV image into a PIL image
    Based on: http://stackoverflow.com/questions/13576161/convert-opencv-image-into-pil-image-in-python-for-use-with-zbar-library
    '''
    cv2_im = cv2.imread(imagePath)
    # Do OpenCV operations
    cv2_im.cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    pil_im.show()

#-------------------------------------------------------------------------------
def pil_to_cv2():
    '''
    Convert a PIL image into an OpenCV image
    Based on: http://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
    '''
    pil_im = PIL.Image.open(imagePath)
    # Do PIL operations
    cv2_im = cv2.cvtColor(numpy.array(pil_im), cv2.COLOR_RGB2BGR)
    cv2.imshow("Image", cv2_im)

#-------------------------------------------------------------------------------
def usage():
    '''Print usage'''
    print("Usage: %s image1 image2 ... imageN" % sys.argv[0])

#-------------------------------------------------------------------------------
def main():
    '''Main function'''
    global prefs
    global stats

    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    prefs = Prefs()
    stats = Stats()

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(prefs.cascade)

    # Detect faces in each image passed in argument
    # Walk through all args and subdirectories
    for path in sys.argv[1:]:
        if os.path.isfile(path):
           detect_faces(path, faceCascade)
           stats.nfiles += 1
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for name in files:
                    if name.endswith('.jpg') or name.endswith('.JPG'):
                        detect_faces(os.path.join(root, name), faceCascade)
                        stats.nfiles += 1
    # Print stats
    print("%d files scanned" % (stats.nfiles))
    print("%d files skipped (too small)" % (stats.skip_image_too_small))
    print("%d faces detected" % (stats.nfaces))
    print("%d faces saved" % (stats.nfaces - stats.skip_face_too_small))
    print("%d faces skipped (too small)" % (stats.skip_face_too_small))

if __name__ == "__main__":
    main()


