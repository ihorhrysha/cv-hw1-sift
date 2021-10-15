

from copy import deepcopy
from typing import List, Tuple
from functools import cmp_to_key
import logging

import numpy as np
from numpy import log, sqrt, floor, round

import cv2
from cv2 import resize, GaussianBlur, subtract, INTER_LINEAR, INTER_NEAREST


# base image sigma
SIFT_INIT_SIGMA = 0.5

# default sigma
SIFT_SIGMA = 1.6

# default number of sampled intervals per octave
SIFT_INTVLS = 3

# default threshold on keypoint ratio of principle curvatures
SIFT_CONTR_THR = 0.04

# default number of bins in histogram for orientation assignment
SIFT_ORI_HIST_BINS = 36

# determines gaussian sigma for orientation assignment
SIFT_ORI_SIG_FCTR = 1.5

# determines the radius of the region used in orientation assignment
SIFT_ORI_RADIUS = 3     # * SIFT_ORI_SIG_FCTR

# orientation magnitude relative to max that results in new feature
SIFT_ORI_PEAK_RATIO = 0.8

# width of border in which to ignore keypoints
SIFT_IMG_BORDER = 5

# default width of descriptor histogram array
SIFT_DESCR_WIDTH = 4

# determines the size of a single descriptor orientation histogram
SIFT_DESCR_SCL_FCTR = 3.

# threshold on magnitude of elements of descriptor vector
SIFT_DESCR_MAG_THR = 0.2

# default number of bins per histogram in descriptor array
SIFT_DESCR_HIST_BINS = 8

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SIFTKeyPoint:
    def __init__(self, pt: Tuple[float, float], size: float = None, angle:float = -1, response: float = 0, octave: int = 0, class_id: int = -1 ) -> None:
        self.pt: Tuple[float, float] = pt
        self.size: float = size
        self.angle: float = angle
        self.response: float = response
        self.octave: int = octave
        self.class_id: int = class_id

    def unpackOctave(self):
        """Compute octave, layer, and scale from a keypoint
        """
        octave = self.octave & 255
        layer = (self.octave >> 8) & 255
        if octave >= 128:
            octave = octave | -128
        scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
        return octave, layer, scale

    def to_cv2(self):
        from cv2 import KeyPoint
        return KeyPoint(
            self.pt[0],
            self.pt[1],
            self.size,
            self.angle,
            self.response,
            self.octave,
            self.class_id
        )

    def convertToOriginalScale(self):

        self.pt = tuple(0.5 * np.array(self.pt))
        self.size *= 0.5
        self.octave = int((self.octave & ~255) | ((self.octave - 1) & 255))
        return self

class SIFTKeyPointsFilter:

    @staticmethod
    def removeDuplicateKeypoints(keypoints: List[SIFTKeyPoint]) -> List[SIFTKeyPoint]:

        if len(keypoints) < 2:
            return keypoints

        def comparator(keypoint1: SIFTKeyPoint, keypoint2: SIFTKeyPoint):
            if keypoint1.pt[0] != keypoint2.pt[0]:
                return keypoint1.pt[0] - keypoint2.pt[0]
            if keypoint1.pt[1] != keypoint2.pt[1]:
                return keypoint1.pt[1] - keypoint2.pt[1]
            if keypoint1.size != keypoint2.size:
                return keypoint2.size - keypoint1.size
            if keypoint1.angle != keypoint2.angle:
                return keypoint1.angle - keypoint2.angle
            if keypoint1.response != keypoint2.response:
                return keypoint2.response - keypoint1.response
            if keypoint1.octave != keypoint2.octave:
                return keypoint2.octave - keypoint1.octave
            return keypoint2.class_id - keypoint1.class_id

        keypoints.sort(key=cmp_to_key(comparator))
        unique_keypoints = [keypoints[0]]

        for next_keypoint in keypoints[1:]:
            last_unique_keypoint = unique_keypoints[-1]
            if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
            last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
            last_unique_keypoint.size != next_keypoint.size or \
            last_unique_keypoint.angle != next_keypoint.angle:
                unique_keypoints.append(next_keypoint)
        return unique_keypoints

    @staticmethod
    def retainBest(keypoints: List[SIFTKeyPoint], num_features:int) -> List[SIFTKeyPoint]:
        return keypoints
    
    @staticmethod
    def runByPixelsMask(keypoints: List[SIFTKeyPoint], mask) -> List[SIFTKeyPoint]:
        raise NotImplementedError

class SIFT:

    def __init__(self, num_features=None, num_octave_intervals=None, contrast_threshold=None, edge_threshold=None, sigma=None) -> None:
             

        self.sigma: float = sigma or SIFT_SIGMA

        self.num_features = num_features
        self.contrast_threshold = contrast_threshold or SIFT_CONTR_THR
        self.num_octave_intervals: int = num_octave_intervals or SIFT_INTVLS
        self.num_octave_images = self.num_octave_intervals + 3 # first + 2 border imgs
        self.edge_threshold = edge_threshold or SIFT_IMG_BORDER
        self.float_tolerance = 1e-7

    
    def _createInitialImage(self, image):       
        if image.ndim == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 2:
            gray_img = cv2.copyTo(image,None)
        else:
            raise ValueError()
        gray_img = image.astype('float32')
        gray_img = resize(gray_img, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
        sig_diff = sqrt(max(self.sigma ** 2 - 4 * (SIFT_INIT_SIGMA ** 2), 0.01))
        return GaussianBlur(gray_img, (0, 0), sigmaX=sig_diff, sigmaY=sig_diff)  # the image blur is now sigma instead of assumed_blur

    # create img pyramids
    def _buildGaussianPyramid(self, image):

        gaussian_pyramid = []
        
        k = 2 ** (1. / self.num_octave_intervals)

        # base image sigma for kernels
        octave_sigmas = np.zeros(self.num_octave_images)
        octave_sigmas[0] = self.sigma

        for image_index in range(1, self.num_octave_images):
            sigma_previous = (k ** (image_index - 1)) * self.sigma
            sigma_total = k * sigma_previous
            octave_sigmas[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)

        
        num_octaves = int(round(log(min(image.shape)) / log(2))) - 1

        for _ in range(num_octaves):
            
            octave_images = []
            octave_images.append(image)
            
            # applying kernel
            for gaussian_kernel in octave_sigmas[1:]:
                image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
                octave_images.append(image)
            
            gaussian_pyramid.append(octave_images)
            
            # last
            base = octave_images[-3]
            image = resize(base, (int(base.shape[1] / 2), int(base.shape[0] / 2)), interpolation=INTER_NEAREST)
        
        return gaussian_pyramid

    def _buildDoGPyramid(self, gaussian_pyramid):

        dog_pyramid = []

        # loop over octaves
        for octave_images in gaussian_pyramid:
            dog_images = []
            for octave_image_idx in range(self.num_octave_images-1):
                octave_image = octave_images[octave_image_idx]
                octave_image_next = octave_images[octave_image_idx+1]
                dog_image = subtract(octave_image_next, octave_image)
                dog_images.append(dog_image)

            dog_pyramid.append(dog_images)

        return dog_pyramid

    # keypoints

    def _isExtrema(self, pixel_cube, threshold):

        center = pixel_cube[1][1, 1]
        if abs(center) > threshold:
            if center > 0:
                return np.all(center >= pixel_cube[0]) and \
                    np.all(center >= pixel_cube[1]) and \
                    np.all(center >= pixel_cube[2])
            elif center < 0:
                return np.all(center <= pixel_cube[0]) and \
                    np.all(center <= pixel_cube[1]) and \
                    np.all(center <= pixel_cube[2])
        return False

    def _adjustLocalExtrema(self, i, j, image_index, octave_index, dog_images_in_octave):

        # constants
        num_attempts_until_convergence=5
        eigenvalue_ratio=10
        
        image_shape = dog_images_in_octave[0].shape
        for _ in range(num_attempts_until_convergence-1):
                       
            top_img, middle_img, bottom_img = dog_images_in_octave[image_index-1:image_index+2]
            pixel_cube = np.stack([top_img[i-1:i+2, j-1:j+2],
                    middle_img[i-1:i+2, j-1:j+2],
                    bottom_img[i-1:i+2, j-1:j+2]]) / 255.

            gradient = self._computeGradient(pixel_cube)
            hessian = self._computeHessian(pixel_cube)
            extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
            if np.all(np.abs(extremum_update)<0.5):
                break
            
            j += int(round(extremum_update[0]))
            i += int(round(extremum_update[1]))
            image_index += int(round(extremum_update[2]))
            # make sure the new pixel_cube will lie entirely within the images
            if i < self.edge_threshold or \
                i >= image_shape[0] - self.edge_threshold or \
                j < self.edge_threshold or \
                j >= image_shape[1] - self.edge_threshold or \
                image_index < 1 or image_index > self.num_octave_intervals:
                return None
        else:
            # ensure convergence
            return None

        functionValueAtUpdatedExtremum = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, extremum_update)
        if abs(functionValueAtUpdatedExtremum) * self.num_octave_intervals >= self.contrast_threshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = np.linalg.det(xy_hessian)
            if xy_hessian_det > 0 and eigenvalue_ratio * (xy_hessian_trace ** 2) < ((eigenvalue_ratio + 1) ** 2) * xy_hessian_det:
                                
                keypoint = SIFTKeyPoint(
                    pt = ((j + extremum_update[0]) * (2 ** octave_index), (i + extremum_update[1]) * (2 ** octave_index)),
                    octave = octave_index + image_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16),
                    size = self.sigma * (2 ** ((image_index + extremum_update[2]) / np.float32(self.num_octave_intervals))) * (2 ** (octave_index + 1)),  # octave_index + 1 because the input image was doubled
                    response = abs(functionValueAtUpdatedExtremum)
                )

                return keypoint, image_index
        return None

    def _computeGradient(self, pixel_cube):
        dx = 0.5 * (pixel_cube[1, 1, 2] - pixel_cube[1, 1, 0])
        dy = 0.5 * (pixel_cube[1, 2, 1] - pixel_cube[1, 0, 1])
        dl = 0.5 * (pixel_cube[2, 1, 1] - pixel_cube[0, 1, 1])
        return np.array([dx, dy, dl])

    def _computeHessian(self, pixel_cube):
        center_pixel_value = pixel_cube[1, 1, 1]
        dxx = pixel_cube[1, 1, 2] - 2 * center_pixel_value + pixel_cube[1, 1, 0]
        dyy = pixel_cube[1, 2, 1] - 2 * center_pixel_value + pixel_cube[1, 0, 1]
        dll = pixel_cube[2, 1, 1] - 2 * center_pixel_value + pixel_cube[0, 1, 1]
        dxy = 0.25 * (pixel_cube[1, 2, 2] - pixel_cube[1, 2, 0] - pixel_cube[1, 0, 2] + pixel_cube[1, 0, 0])
        dxl = 0.25 * (pixel_cube[2, 1, 2] - pixel_cube[2, 1, 0] - pixel_cube[0, 1, 2] + pixel_cube[0, 1, 0])
        dyl = 0.25 * (pixel_cube[2, 2, 1] - pixel_cube[2, 0, 1] - pixel_cube[0, 2, 1] + pixel_cube[0, 0, 1])
        return np.array([[dxx, dxy, dxl], 
                    [dxy, dyy, dyl],
                    [dxl, dyl, dll]])

    def _computeOrientationHistogram(self, image, keypoint: SIFTKeyPoint, octave_index: int):
        
        image_height, image_width = image.shape

        # constants
        radius_factor=SIFT_ORI_RADIUS
        num_bins=SIFT_ORI_HIST_BINS 
        scale_factor=SIFT_ORI_SIG_FCTR

        scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))
        radius = int(round(radius_factor * scale))
        weight_factor = -0.5 / (scale ** 2)

        # computing histogram
        raw_histogram = np.zeros(num_bins)
        
        for i in range(-radius, radius + 1):
            region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
            if region_y > 0 and region_y < image_height - 1:
                for j in range(-radius, radius + 1):
                    region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                    if region_x > 0 and region_x < image_width - 1:
                        dx = image[region_y, region_x + 1] - image[region_y, region_x - 1]
                        dy = image[region_y - 1, region_x] - image[region_y + 1, region_x]
                        gradient_magnitude = sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2)) 
                        histogram_index = int(round(gradient_orientation * num_bins / 360.))
                        raw_histogram[histogram_index % num_bins] += weight * gradient_magnitude

        # histogram smoothing
        smooth_histogram = np.zeros(num_bins)
        for n in range(num_bins):
            smooth_histogram[n] = (6 * raw_histogram[n] + 4 * (raw_histogram[n - 1] + raw_histogram[(n + 1) % num_bins]) + raw_histogram[n - 2] + raw_histogram[(n + 2) % num_bins]) / 16.
        
        return smooth_histogram

    def _computeKeypointsWithOrientations(self, image, keypoint: SIFTKeyPoint, octave_index: int) -> List[SIFTKeyPoint]:
        
        keypoints = []
        
        num_bins=SIFT_ORI_HIST_BINS 
        peak_ratio=SIFT_ORI_PEAK_RATIO
        
        smooth_histogram = self._computeOrientationHistogram(image=image, keypoint=keypoint, octave_index=octave_index)
               
        orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]
        orientation_max = max(smooth_histogram)
        for peak_index in orientation_peaks:
            peak_value = smooth_histogram[peak_index]
            
            # check peeks threshold
            if peak_value < peak_ratio * orientation_max:
                continue

            # peak interpolation
            left_value = smooth_histogram[(peak_index - 1) % num_bins]
            right_value = smooth_histogram[(peak_index + 1) % num_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % num_bins
            angle = 360. - interpolated_peak_index * 360. / num_bins
            angle = angle if abs(angle - 360.) >= self.float_tolerance else 0
                                        
            # creating oriented point
            new_keypoint: SIFTKeyPoint = deepcopy(keypoint)
            new_keypoint.angle = angle
            new_keypoint.convertToOriginalScale()

            keypoints.append(new_keypoint)

        return keypoints

    def _filterKeypoints(self, keypoints: List[SIFTKeyPoint], mask=None) -> List[SIFTKeyPoint]:
        
        keypoints = SIFTKeyPointsFilter.removeDuplicateKeypoints(keypoints)
        
        if self.num_features:
            keypoints = SIFTKeyPointsFilter.retainBest(keypoints, num_features=self.num_features)

        if mask:
            keypoints = SIFTKeyPointsFilter.runByPixelsMask(keypoints=keypoints, mask=mask)

        return keypoints

    def _get_pixel_cube(self, i, j, top_img, middle_img, bottom_img):
        slicer = (slice(i-1,i+2), slice(j-1,j+2))

        # simple array is more efficient/ no copies in memory just pointers
        return [top_img[slicer],
                middle_img[slicer],
                bottom_img[slicer]]

    def _findKeypoints(self, gaussian_pyramid, dog_pyramid):

        threshold = floor(0.5 * self.contrast_threshold / self.num_octave_intervals * 255)
        keypoints = []

        # loop over octaves
        for octave_idx, dog_images in enumerate(dog_pyramid):

            # loop over triplets of layers
            for interval_idx in range(self.num_octave_intervals):
                top_img = dog_images[interval_idx]
                middle_img = dog_images[interval_idx+1]
                bottom_img = dog_images[interval_idx+2]
                 
                image_height, image_width = middle_img.shape 

                # (i, j) is the center of the 3x3 array in middle image
                for i in range(self.edge_threshold, image_height - self.edge_threshold):
                    for j in range(self.edge_threshold, image_width - self.edge_threshold):
                        
                        pixel_cube = self._get_pixel_cube(i, j, top_img, middle_img, bottom_img)
                        # check middle pixel
                        if not self._isExtrema(pixel_cube, threshold):
                            continue

                        localization_result = self._adjustLocalExtrema(i, j, interval_idx + 1, octave_idx, dog_images)
                        if not localization_result:
                            continue

                        keypoint, localized_image_index = localization_result
                        keypoints_with_angle = self._computeKeypointsWithOrientations(
                            gaussian_pyramid[octave_idx][localized_image_index], 
                            keypoint, 
                            octave_idx
                        )
                        keypoints.extend(keypoints_with_angle)

        return keypoints

    # TODO optimize
    # descriptors from keypoints
    def calcDescriptors(self, keypoints: List[SIFTKeyPoint], gaussian_images):
        

        window_width=SIFT_DESCR_WIDTH
        num_bins=SIFT_DESCR_HIST_BINS
        scale_multiplier=SIFT_DESCR_SCL_FCTR
        descriptor_max_value=SIFT_DESCR_MAG_THR

        descriptors = []

        for keypoint in keypoints:
            octave, layer, scale = keypoint.unpackOctave()
            gaussian_image = gaussian_images[octave + 1][layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   # first two dimensions are increased by 2 to account for border effects

            # Descriptor window size (described by half_width) follows OpenCV convention
            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            half_width = int(round(hist_width * sqrt(2) * (window_width + 1) * 0.5))   # sqrt(2) corresponds to diagonal length of a pixel
            half_width = int(min(half_width, sqrt(num_rows ** 2 + num_cols ** 2)))     # ensure half_width lies within image

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(round(point[1] + row))
                        window_col = int(round(point[0] + col))
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                # Smoothing via trilinear interpolation
                # Notations follows https://en.wikipedia.org/wiki/Trilinear_interpolation
                # Note that we are really doing the inverse of trilinear interpolation here (we take the center value of the cube and distribute it among its eight neighbors)
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

            descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
            # Threshold and normalize descriptor_vector
            threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), self.float_tolerance)
            # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
            descriptor_vector = np.round(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype='float32')

    def detectAndCompute(self, image, mask=None, keypoints=None):
       
        useProvidedKeypoints = keypoints is not None

        base = self._createInitialImage(image)
        gaussian_pyramid = self._buildGaussianPyramid(base)
        
        if not useProvidedKeypoints:
            dog_pyramid = self._buildDoGPyramid(gaussian_pyramid)
            
            keypoints = self._findKeypoints(gaussian_pyramid, dog_pyramid)
            keypoints = self._filterKeypoints(keypoints=keypoints, mask=mask)

        descriptors = self.calcDescriptors(keypoints, gaussian_pyramid)
        return keypoints, descriptors

    def compute(self, image, keypoints):
        # for benchmark
        return self.detectAndCompute(image, keypoints=keypoints)

def SIFT_create(num_features=None, num_octave_intervals=None, contrast_threshold=None, edge_threshold=None, sigma=None) -> SIFT:
    return SIFT(num_features=num_features, num_octave_intervals=num_octave_intervals, contrast_threshold=contrast_threshold, edge_threshold=edge_threshold, sigma=sigma)


if __name__ == '__main__':

    img1 = cv2.imread('./res/lenna_grey_cropped.png', 0)
    
    sift = SIFT_create()

    # Compute SIFT keypoints and descriptors
    kp1, desc1 = sift.detectAndCompute(img1)