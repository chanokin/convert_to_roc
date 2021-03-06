import numpy
# from scipy.signal import sepfir2d, convolve2d
from cv2 import sepFilter2D
import cv2

DISABLED_PIXEL_VAL = float(-10e9)
class Convolution():
    '''
    Utility class to wrap around different functions for convolution 
    (i.e. separable convolution)
    '''


    def __init__(self, disabled_pixel_value=DISABLED_PIXEL_VAL):
        self.disabled_pixel_value = disabled_pixel_value

    def sep_convolution(self, img, horz_k, vert_k, col_keep=1, row_keep=1, mode="full",
                      fill=0):
        ''' Separated convolution -
            img      => image to convolve
            horiz_k  => first convolution kernel vector (horizontal)
            vert_k   => second convolution kernel vector (horizontal)
            col_keep => which columns are we supposed to calculate
            row_keep => which rows are we supposed to calculate
            mode     => if "full": convolve all the image otherwise just valid pixels
            fill     => what fill value to use 
        '''
        width  = img.shape[1]
        height = img.shape[0]
        half_k_width = horz_k.size//2
        half_img_width  = width//2
        half_img_height = height//2

        tmp = numpy.zeros_like(img, dtype=numpy.float32)

        if mode == "full":
            horizontal_range = numpy.arange(width)
            vertical_range   = numpy.arange(height)
        else:
            s = max(half_k_width//2, 1)
            horizontal_range = numpy.arange(s, width  - s + 1)
            vertical_range   = numpy.arange(s, height - s + 1)

        for y in numpy.arange(height):
            for x in horizontal_range:
                if (x - half_img_width)%col_keep != 0:
                    continue

                k_sum = 0.
                k = 0

                for i in numpy.arange(-half_k_width, half_k_width + 1):
                    img_idx = x + i
                    if img_idx >= 0 and img_idx < width:
                        k_sum += img[y, img_idx]*horz_k[k]
                    else:
                        k_sum += fill * horz_k[k]
                    k += 1

                tmp[y,x] = k_sum

        tmp2 = numpy.zeros_like(img, dtype='float32')
        for y in vertical_range:
            if (y - half_img_height)%row_keep != 0:
                continue

            for x in horizontal_range:
                if (x - half_img_width)%col_keep != 0:
                    continue

                k_sum = 0.
                k = 0
                for i in numpy.arange(-half_k_width, half_k_width + 1):
                    img_idx = y + i
                    if img_idx >= 0 and img_idx < height:
                        k_sum += tmp[img_idx, x]*vert_k[k]
                    else:
                        k_sum += fill * vert_k[k]
                    
                    k += 1

                tmp2[y,x] = k_sum

        return tmp2


    def dog_sep_convolution(self, img, k, cell_type, originating_function="filter",
                           force_homebrew = False, mode="full", is_off_center=False):
        ''' Wrapper for separated convolution for DoG kernels in FoCal, 
            enables use of NumPy based sepfir2d.
            
            img                  => the image to convolve
            k                    => 1D kernels to use
            cell_type            => ganglion cell type, useful for sampling 
                                    resolution numbers
            originating_function => if "filter": use special sampling resolution,
                                    else: use every pixel
            force_hombrew        => if True: use my code, else: NumPy's
            mode                 => "full" all image convolution, else only valid
        '''

        fill = 0.0 if is_off_center else 0.0

        if originating_function == "filter":
            row_keep, col_keep = self.get_subsample_keepers(cell_type)
        else:
            row_keep, col_keep = 1, 1


        # if not force_homebrew :
        # if cell_type in [0, 1]:
            # has a problem with images smaller than kernel
            
            # center_img = sepfir2d(img.copy(), k[0], k[1])#, mode='same', cval=fill)
            # surround_img  = sepfir2d(img.copy(), k[2], k[3])#, mode='same', cval=fill)
            # center_img = sepFilter2D(img.copy(), -1, k[0], k[1], 
            #                 borderType=cv2.BORDER_REFLECT_101)
            # surround_img  = sepFilter2D(img.copy(), -1, k[2], k[3],
            #                     borderType=cv2.BORDER_REFLECT_101)

        # else:
        if True:
            center_img = self.sep_convolution(img.copy(), k[0], k[1],
                            col_keep=col_keep, row_keep=row_keep,
                            mode='valid', fill=fill)
            surround_img  = self.sep_convolution(img.copy(), k[2], k[3],
                            col_keep=col_keep, row_keep=row_keep,
                            mode='valid', fill=fill)

        conv_img = center_img + surround_img

        # # normalize so auto-correlation == 1 ?
        # conv_img *= 1./numpy.sqrt(numpy.sum(conv_img**2))

        if not force_homebrew and originating_function == "filter":
            conv_img = self.subsample(conv_img, cell_type)
            # conv_img[:] = numpy.clip(conv_img, 0.0, numpy.inf)

        return conv_img


    def get_subsample_keepers(self, cell_type):
        ''' return which (modulo) columns and rows to keep for cell_type
        '''
        if cell_type == 3:
            # col_keep = 7
            # row_keep = 7
            col_keep = 3
            row_keep = 3
        elif cell_type == 2:
            # col_keep = 5
            # row_keep = 3
            col_keep = 3
            row_keep = 3
        elif cell_type == 1:
            col_keep = 1
            row_keep = 1
        else:
            col_keep = 1
            row_keep = 1

        return row_keep, col_keep


    def subsample(self, img, cell_type):
        ''' remove unwanted rows/columns '''
        row_keep, col_keep = self.get_subsample_keepers(cell_type)
        
        if col_keep < img.shape[1] and row_keep < img.shape[0]:
            width = img.shape[1]
            height = img.shape[0]
            half_img_width  = width//2
            half_img_height = height//2
            
            col_range = [x for x in numpy.arange(width)
                         if (x - half_img_width)%(col_keep) != 0]
            row_range = [x for x in numpy.arange(height)
                         if (x - half_img_height)%(row_keep) != 0]
            
            img[:, col_range] = self.disabled_pixel_value
            img[row_range, :] = self.disabled_pixel_value
            #~ img[:, [x for x in col_range if (x)%(col_keep)!= 0]] = 0
            #~ img[[x for x in row_range if (x)%(row_keep)!= 0], :] = 0
        else:
            img[:,:] = 0

        return img
