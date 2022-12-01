
import numpy as np
import cv2 as cv

def birdseye_view(img: np.ndarray, H: np.ndarray) -> np.ndarray:
    """apply perspective transformation to obtain bird's eye view"""
    # warp the image using precalculated homography matrix 
    cuda_img = cv.cuda_GpuMat(img)
    # cuda_H = cv.cuda_GpuMat(H)

    height = img.shape[0]
    width = img.shape[1]
    transformed_frame = cv.cuda.warpPerspective(cuda_img, H, (width, height))

    # apply a downward translation (temporarily)
    T = np.array([[1, 0, 100], 
                [0, 1, 100]], dtype=np.float32)

    translated_frame = cv.cuda.warpAffine(transformed_frame, T, (width, height))

    return translated_frame
