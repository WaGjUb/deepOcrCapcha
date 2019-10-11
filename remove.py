import cv2
import numpy as np

# Sort contour by size 
def sort_contour(contours):

    ret = []
    d = {}
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = h*w 
        try:
            d[area].append(cnt)
        except:
            d[area] = [cnt]

    keys = list(d.keys())
    keys.sort(reverse = True)

    for k in keys:
        ret = ret + d[k]
    return ret

        
# remove background
## im is an gray image 
def bg_remove(im):
    y,x = im.shape
    
    # get background color for each quadrant
    mq1 = np.median(im[0:int(y/2), 0:int(x/2)])
    mq2 = np.median(im[0:int(y/2), int(x/2):])
    mq3 = np.median(im[int(y/2):, 0:int(x/2)])
    mq4 = np.median(im[int(y/2):, int(x/2):])
    medians = [mq1, mq2, mq3]#, mq4]

    # create masks
    soma = np.zeros((y,x), np.uint8)
    for m in medians:
        mask = cv2.inRange(im, m-10, m+10)
        if (type(soma) == None):
            soma = mask
        else:
            soma += mask
    cv2.imshow("soma", soma)
    cv2.waitKey(0)
    
    ret_img = soma
  
    return ret_img 


for i in range(100):
    img = cv2.imread("captchas/{}.jpg".format(i))
    # take off the borders
    img = img[1:-1, 1:-1]
    # blur to remove line
    img2 = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # remove background
    without_bg = bg_remove(gray)
    # get contours 
    edges = cv2.Canny(without_bg, 200, 200)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sort_contour(contours)
    y,x = gray.shape
    blank = np.full((y,x),255, np.uint8)

    # draw 6 contours with an thick line in an blank image
    for idx, cnt in enumerate(sorted_contours):
        if idx < 10:
            #cont_area = cv2.contourArea(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(blank, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # get conturs again to merge some bounding boxes 
    edges = cv2.Canny(blank, 200, 200)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sort_contour(contours)
    # draw 5 biggest contours
    for idx, cnt in enumerate(sorted_contours):
        #cont_area = cv2.contourArea(cnt)
        if idx < 5:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    cv2.imshow("blank", gray)
    cv2.imshow("edges", edges)
    cv2.imshow('final', img)


    cv2.waitKey(0)


