# adjust levels to fill histogram
# Kevin Cazabon, 1999.  kcazabon at rogers.wave.ca  kcaza at cymbolic.com
#
# https://mail.python.org/pipermail/image-sig/2005-September/003514.html
# http://code.activestate.com/lists/python-image-sig/%3cs6c95fdd.013@cymbolic.com%3e/


def levels(data, all_same = 1, clip = 0):
    """Adjust image levels

    - checks the histogram, determines the actual brightness range used
      in the image for each channel
    - scales the data to fill the full 0-255 range either by scaling
      each channel individually, or by scaling them equally (to the
      lowest common denomonator)

    :param all_same: 0 = scale each channel individually, 1 = scale all
                     channels the same
    :param clip: integer, if there's less than this number of pixels at
                 the minimum/maximum brightness level found, don't use
                 them to calculate the levels adjustment (i.e. they get
                 clipped off the image to pure black/white)  default is
                 0.  This is useful if you have a few 'dark' or 'bright'
                 pixels that actually are not relevant detail and are
                 causing the scaling to not properly fill up the
                 histogram.
    """
    if data.mode not in ['RGB', 'CMYK']:
        return data

    lut = makelut(data, all_same, clip)

    data = data.point(lut)

    return data

def find_hi_lo(lut, clip):
    min = None
    max = None

    for i in range(len(lut)):
        if lut[i] > clip:
            min = i
            break

    lut.reverse()

    for i in range(len(lut)):
        if lut[i] > clip:
            max = 255 - i
            break

    return min, max

def scale(channels, min, max):
    lut = []
    for i in range (channels):
        for i in range(256):
            value = int((i - min)*(255.0/float(max-min)))
            if value < 0:
                value = 0
            if value > 255:
                value = 255
            lut.append(value)

    return lut


def makelut(data, all_same, clip):
    import PIL.Image

    histogram = data.histogram()

    lut = []
    r, g, b, k = [], [], [], []

    channels = len(histogram) // 256

    for i in range(256):
        r.append(histogram[i])
        g.append(histogram[256+i])
        b.append(histogram[512+i])
    if channels == 4:
        for i in range(256):
            k.append(histogram[768+i])


    rmin, rmax = find_hi_lo(r, clip)
    gmin, gmax = find_hi_lo(g, clip)
    bmin, bmax = find_hi_lo(b, clip)
    if channels == 4:
        kmin, kmax = find_hi_lo(k)
    else:
        kmin, kmax = 128, 128

    if all_same == 1:

        min_max = [rmin, gmin, bmin, kmin, rmax, gmax, bmax, kmax]
        min_max.sort()
        lut = scale(channels, min_max[0], min_max[-1])

    else:

        r_lut = scale(1, rmin, rmax)
        g_lut = scale(1, gmin, gmax)
        b_lut = scale(1, bmin, bmax)
        if channels == 4:
            k_lut = scale(1, kmin, kmax)

        lut = []

        for i in range (256):
            lut.append(r_lut[i])
        for i in range (256):
            lut.append(g_lut[i])
        for i in range (256):
            lut.append(b_lut[i])
        if channels == 4:
            for i in range (256):
                lut.append(k_lut[i])

    return lut
