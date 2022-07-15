import easygui
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import math

def click_event(event, x,y,flags,params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        x_coords.append(x)
        y_coords.append(y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.drawMarker(displayimg, (x,y),(0,0,255),markerType=cv2.MARKER_CROSS, markerSize=200, thickness=10, line_type=cv2.LINE_AA)
        cv2.imshow('Image', displayimg)

def display_image(img):
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def main():
    # file = easygui.fileopenbox()
    file = 'images/TN-002-991_s1(1).tif'
    im = cv2.imread(file)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image',im)
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #Takes in the image and displays it, with a crosshair to indicate where the user can click to set the crop coordinates
    file = 'images/TN-002-991_s1(1).tif'
    img = cv2.imread(file)
    displayimg = img.copy()
    x_coords = []
    y_coords = []
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image',displayimg)
    cv2.setMouseCallback('Image', click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    x_1 = 0
    x_2 = 0
    y_1 = 0
    y_2 = 0
    if x_coords[0] < x_coords[1]:
            x_1 = x_coords[0]
            x_2 = x_coords[1]
    else:
        x_2 = x_coords[0]
        x_1 = x_coords[1]

    if y_coords[0] < y_coords[1]:
        y_1 = y_coords[0]
        y_2 = y_coords[1]
    else:
        y_2 = y_coords[0]
        y_1 = y_coords[1]

    # The image is cropped and saved locally as im_cropped
    im_cropped = img[y_1:y_2, x_1:x_2]
    im_cropped_gray = cv2.cvtColor(im_cropped,cv2.COLOR_BGR2GRAY)
    im_cropped_gray = cv2.blur(im_cropped_gray,(20,20))
    thresh, im_cropped_bw = cv2.threshold(im_cropped_gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(im_cropped_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        print(len(contour))

    cv2.drawContours(im_cropped, contours, -1, (0,255,0), 3)
    cv2.namedWindow('Cropped', cv2.WINDOW_NORMAL)
    cv2.imshow('Cropped', im_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #
    # # cnt,_ = cv2.findContours(im_cropped_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # img_cnt = np.zeros(im_cropped_bw.shape)
    # # cv2.drawContours(img_cnt, cnt, -1, (0,255,0),3)
    #
    #
    # # coords = np.column_stack(np.where(im_cropped_bw > 1))
    # # rect = cv2.minAreaRect(coords)
    # # box = cv2.boxPoints(rect)
    # # box = np.int0(box)
    im_cropped_contours = im_cropped.copy()
    contours = np.vstack(contours).squeeze()
    rect = cv2.minAreaRect(contours)
    box = cv2.boxPoints(rect)
    angle = rect[2]
    box = np.int0(box)
    cv2.drawContours(im_cropped_contours,[box],0,(0,0,255),2)
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    cv2.imshow('Contours', im_cropped_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(angle)
    image_center = tuple(np.array(img.shape[1::-1])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle+90,1)
    result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    cv2.namedWindow('Rotated', cv2.WINDOW_NORMAL)
    cv2.imshow('Rotated', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('images/rotated.tif', result)

