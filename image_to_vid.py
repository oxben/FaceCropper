#!/usr/bin/env python

"""
Convert a set of images to a video with blending transitions between each input image

Author: Oxben <oxben@free.fr>
"""

import logging as logging
import os.path
import shutil
import subprocess
import sys
import tempfile

log = logging.getLogger(__file__)

def exec_cmd(cmd, wait=True):
    '''Run an external command'''
    ret = 0
    process = subprocess.Popen(cmd.split(),
                                shell=False,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if stderr is None or stderr is '':
        ret = 0
    if wait:
        ret = process.wait()
    return ret, stdout, stderr

class Prefs:

    def __init__(self):
        self.temp_dir = '/tmp/'
        self.morph_out_format = 'morph-%03d.jpg'
        self.morph_duration = 0.5
        self.keyframe_duration = 1
        self.frame_prefix = 'img'
        self.frame_rate = 24
        self.video_file = 'cam.mp4'

class ImageToVideo:

    def __init__(self):
        self.prefs = Prefs()
        self.temp_dir = None
        self.images = None
        self.frame_counter = 0

    def usage(self):
        print('{0}: image1 image2 ...'.format(sys.argv[0]))

    def cleanup(self):
        """Remove all temporary resources"""
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)

    def multicopy(self, srcfile):
        """Add multiple copies of specified image to output directory"""
        frames = self.prefs.keyframe_duration * self.prefs.frame_rate
        for i in range(frames):
            dstfile = '{}-{:04d}.jpg'.format(self.prefs.frame_prefix, self.frame_counter)
            shutil.copy(srcfile, os.path.join(self.temp_dir, dstfile))
            self.frame_counter += 1

    def run(self, images):
        """Run images conversion"""
        if len(images) < 2:
            usage()
            return 1

        self.images = images

        # Create temporary working directory
        self.temp_dir = tempfile.mkdtemp(dir=self.prefs.temp_dir)

        log.info('Calculate inbetweens')
        prev_img = None
        for curr_img in self.images:

            if prev_img == None:
                # First image, no morph, only duplicate key frame
                self.multicopy(curr_img)
                prev_img = curr_img
                continue

            # Create intermediate frames
            frames = int(self.prefs.morph_duration * self.prefs.frame_rate)
            cmd = 'convert {0} {1} -morph {2} {3}'.format(prev_img, \
                                                       curr_img, \
                                                       frames, \
                                                       os.path.join(self.temp_dir, self.prefs.morph_out_format))
            ret, out, err = exec_cmd(cmd)
            log.debug('{0} : ret={1} out={2} err={3}'.format(cmd, ret, out, err))

            # Rename inbetweens images
            total_frames = frames + 2
            for i in range(0, total_frames):
                srcfile = os.path.join(self.temp_dir, self.prefs.morph_out_format % i)
                dstfile = os.path.join(self.temp_dir, '{}-{:04d}.jpg'.format(self.prefs.frame_prefix, self.frame_counter))
                if i == 0:
                    # Skip first morph image (key frame)
                    os.remove(srcfile)
                elif i == total_frames - 1:
                    # Copy last morph image (key frame)
                    self.multicopy(srcfile)
                    os.remove(srcfile)
                else:
                    # Rename inbetweens
                    log.debug('{0} -> {1}'.format(srcfile, dstfile))
                    os.rename(srcfile, dstfile)
                    self.frame_counter += 1

        # Encode video
        log.info('Encode video: {}'.format(self.prefs.video_file))
        img_format = r'{}-%04d.jpg'.format(self.prefs.frame_prefix)
        cmd = 'ffmpeg -r {} -f image2 -i {} -s 256x256 {}'.format( \
                    self.prefs.frame_rate, \
                    os.path.join(self.temp_dir, img_format), \
                    self.prefs.video_file)
        ret, out, err = exec_cmd(cmd, False)
        log.debug('{0} : ret={1} out={2} err={3}'.format(cmd, ret, out, err))

        # Destroy temporary resources
        self.cleanup()
        log.info('Done')

def main():
    logging.basicConfig(level=logging.INFO)
    app = ImageToVideo()
    rc = app.run(sys.argv[1:])
    sys.exit(rc)

if __name__ == "__main__":
    main()
