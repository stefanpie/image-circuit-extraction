import numpy as np
import cv2


def line_detecion(img):

    img_out_s1 = np.zeros(img.shape, dtype='uint8')
    img_out_s2 = np.zeros(img.shape, dtype='uint8')
    img_out_s3 = np.zeros(img.shape, dtype='uint8')
    dst = np.copy(img)

    #################
    # 1st HoughP Stage
    ################
    maxVal = 255
    minVal = maxVal * 0.7
    AccThreshold = 60
    MinLine = 30
    MaxGap = 10

    # Find edges using the Canny method
    gaussian_s1 = cv2.GaussianBlur(img, (3, 3), 1)
    canny_s1 = cv2.Canny(gaussian_s1, threshold1=minVal, threshold2=maxVal)
    dilate_s1 = cv2.dilate(canny_s1, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    lines_s1 = cv2.HoughLinesP(dilate_s1, 2, np.pi / 90.0, AccThreshold, minLineLength=MinLine, maxLineGap=MaxGap)

    # Draw lines on the detected points
    for line1 in lines_s1:
        x1, y1, x2, y2 = line1[0]
        cv2.line(img_out_s1, (x1, y1), (x2, y2), (255, 255, 255), 1)

    #################
    # 2nd HoughP Stage
    ################
    img_out_s1_reopen = img_out_s1
    CannyMaxVal_s2 = 255
    CannyMinVal_s2 = CannyMaxVal_s2*0.7
    HoughPThres_s2 = 40
    MinLine_s2 = 20
    MaxGap_s2 = 10

    dilate_s2 = cv2.dilate(img_out_s1_reopen, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    kernel_s2 = np.ones((5, 5), np.uint8)
    erosion_s2 = cv2.erode(dilate_s2, kernel_s2, iterations=1)
    lines_s2 = cv2.HoughLinesP(erosion_s2[:,:,0], 2, np.pi / 90.0, HoughPThres_s2, minLineLength=MinLine_s2, maxLineGap=MaxGap_s2)

    # Draw lines on the detected points
    for line2 in lines_s2:
        x1, y1, x2, y2 = line2[0]
        cv2.line(img_out_s2, (x1, y1), (x2, y2), (255, 255, 255), 1)

    #################
    # 3rd HoughP Stage
    ################
    img_out_s2_reopen = img_out_s2
    CannyMaxVal_s3 = 255
    CannyMinVal_s3 = CannyMaxVal_s3 * 0.7
    HoughPThres_s3 = 40
    MinLine_s3 = 20
    MaxGap_s3 = 10

    lines_s3 = cv2.HoughLinesP(img_out_s2_reopen[:,:,0], 3, np.pi / 90.0, HoughPThres_s3, minLineLength=MinLine_s3, maxLineGap=MaxGap_s3)

    # Draw lines on the detected points
    for line3 in lines_s3:
        x1, y1, x2, y2 = line3[0]
        cv2.line(img_out_s3, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(dst, (x1, y1), (x2, y2), (255, 255, 255), 3)
    
    img_out_s3 = cv2.threshold(img_out_s3, 15, 255, cv2.THRESH_BINARY)[1]

    return img_out_s3
