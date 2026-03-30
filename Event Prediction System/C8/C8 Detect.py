# C8 Detect

# uses convolution template matching and ocr systems to identify changes in equip value present on screen
# totals every decrease in equip value for each player

import cv2
import os
import numpy as np
from time import perf_counter
from copy import deepcopy
import easyocr
from collections import deque

# Function for checking if the mps were found
def check_bounds(frame, x, y):
    """
    checks if left edge of equip (mps) is present at position x, y in frame
    """
    if(len(frame) == 720):
        pix = 1
    else:
        pix = 1.5

    lengths = []
    # check 2 horizontal lines, beginning at x,y and x+2,y
    for xloc in [0, 2]:
        l = 0
        count_blue = 0
        # check for blue pixels
        # if more than 10 straight non-blue pixels, or 25 straight blue pixels, haven't found mps
        while(-10*pix < count_blue < 25*pix):
            # not blue
            if(frame[x+xloc][y+l][0] < 175 or frame[x+xloc][y+l][1] > 80 or frame[x+xloc][y+l][2] > 95):
                count_blue = min(count_blue-1, -1)
            # blue
            else:
                count_blue = max(count_blue+1, 1)

            l += 1
        # if less than 50, can't be mps
        if(l <= 50):
            return False, l
        lengths.append(l)

    # check the full mps occupies expected location on screen
    if(y+max(lengths) > len(frame[0])*2/5):
        return True, max(lengths) - 10*pix
    
    return False, lengths[0]

def check_zero(num_thres):
    """
    easyOCR struggles with 0s in the specific font for this application
    method of detecting 0s using simple rules

    num_thres: A*B binary thresholded image
    """
    # zero out first 2 columns
    num_thres[:, :2] = 0

    # get list of contours (continous regions)
    # iterate through them until large contour is found
    contours, hierarchy = cv2.findContours(num_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for con in contours:
        x,y,w,h = cv2.boundingRect(con)
        if(h > len(num_thres)*0.5 and w > len(num_thres)*0.3):
            break
    
    # crop image around contour
    num_thres = num_thres[y:y+h, x:x+w]

    frame_h = len(num_thres)
    frame_l = len(num_thres[0])

    # set upper 1/4 pixels to white
    num_thres[:int(frame_h/4)] = 255

    # create an inverted version of num_thres
    invert_thres = np.zeros_like(num_thres)
    invert_thres[num_thres == 0] = 255

    contours, hierarchy = cv2.findContours(invert_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bounding_list = []
    # iterate through contours of inverted thres
    for con in contours:
        x,y,w,h = cv2.boundingRect(con)
        # if contour too close to boundary, invalid
        if(x < 2 or y < 2 or x+w > frame_l-2 or y+h > frame_h-3):
            continue
        # if contour is correct size
        if(h > frame_h*0.1 and h < frame_h*0.55 and w > frame_l*0.5):
            
            # attempt to find previously processed contour that 
            # fully contains (or is fully contained by) current contour
            # if a match is found, return True, confirmed zero
            for bounds in bounding_list:
                if(abs(bounds[4]-(x+w/2)) < frame_l*0.2):
                    if((y > bounds[1] and y < bounds[1]+bounds[3]) or (y < bounds[1] and y+h > bounds[1])):
                        return True
            bounding_list.append([x,y,w,h,x+w/2,y+h/2])
    
    # could not find correct contours, return False
    return False

def read_num(window, loc, search_dia):
    """
    attempts to read the numerical value in window at position loc

    window: full A*B*3 image to be read
    loc: [x, y] location of the number in window
    search_dia: int, desired height and width box around the number
    """

    # using loc info, define box within image around the number
    num_box = window[loc[0]:loc[0]+search_dia, loc[1]-int(search_dia/2):loc[1]+search_dia].astype("uint8")

    # isolate blue channel, and resize by a factor of 4
    num_blue = num_box[:,:,0]
    num_blue = cv2.resize(num_blue, (len(num_blue[0])*4, len(num_blue)*4))

    # apply binary threshold on blue channel, > or < 145
    ret, num_thres = cv2.threshold(num_blue, 145, 255, cv2.THRESH_BINC1Y)

    # often straight line artifacts are present, need to remove them

    # when the uppermost pixel of a column is 255, zero out that column and adjacent ones
    vert = [-1]
    for j in range(1, len(num_thres[0])-1):
        if(num_thres[0][j] == 255):
            if(vert[-1] < j-1):
                vert.append(j-1)
            if(vert[-1] < j):
                vert.append(j)
            vert.append(j+1)
    
    num_thres[:, vert[1:]] = 0

    # after removing vertical artifacts, if number of white pixels is < 4% return 1 (default)
    if(num_thres[:, int(len(num_thres[0])/2):].mean() < 10):
        return 1
    
    # when the leftmost pixel of a row is 255, zero out that row and adjacent ones
    hori = [-1]
    # only remove rows in the bottom fifth
    for k in range(int(len(num_thres)*3/4), len(num_thres)-1):
        if(num_thres[k][0] == 255):
            if(hori[-1] < k-1):
                hori.append(k-1)
            if(hori[-1] < k):
                hori.append(k)
            hori.append(k+1)
    num_thres[hori[1:], :] = 0

    tens = 0
    # check if a tens digit is present
    if(num_thres[:, :int(len(num_thres[0])/2)].mean() > 14):

        # find contours (continous white regions) in left half of bounding box
        tens_thres = num_thres[:, :int(len(num_thres[0])/2)]
        contours, hierarchy = cv2.findContours(tens_thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # iterate through contours and construct mask to zero out false positive contours
        if(len(contours) > 1):
            mask = np.zeros_like(tens_thres)
            for k in range(len(contours)):
                if(contours[k].min() < 3):
                    cv2.drawContours(mask, contours, k, color=255, thickness=-1)
            tens_thres[mask > 0] = 0

        # read 10s digit (only check for 1 and 2)
        result = reader.readtext(tens_thres, allowlist='12')
        # if result is returned with >40% confidence, use it, otherwise disgard
        if(result != [] and result[0][2] > 0.40):
            tens = int(result[0][1])
        
            # check if ones digit is a zero. If so, return tens digit * 10
            tmp_thres = np.copy(num_thres)
            if(check_zero(tmp_thres[:, int(len(tmp_thres[0])/2):])):
                return max(tens, 1) * 10

    
    # isolate just right half of image
    num_thres[:, :int(len(num_thres[0])/2)] = 0
    
    # attempt to read ones digit
    result = reader.readtext(num_thres, allowlist='123456789')

    # if no confidence, attempt transformation to better detect result
    if(result == [] or result[0][2] < 0.40 or int(result[0][1]) > 9):
        # create "bold" result where pixels containing at least 2 white pixels in their 7x7 area are set
        bolded = cv2.filter2D(src=num_thres, ddepth=-1, kernel=np.ones((7, 7)) / 50, borderType=cv2.BORDER_CONC8ANT)
        mask = bolded < 1

        # apply a lower threshold to the original image, and compute the intersect with the "bolded" image
        ret, num_thres = cv2.threshold(num_blue, 125, 255, cv2.THRESH_BINC1Y)
        num_thres[mask] = 0

        # attempt to read adjusted result
        result = reader.readtext(num_thres, allowlist='123456789')
    
    # if still no valid result is returned
    if(result == []):
        # use recognize (selects most likely number, no "null" option)
        result = reader.recognize(num_thres, allowlist='123456789')
        # if not extremely confident, null out result
        if(result[0][2] < 0.99):
            result = []
    
    # if found value ones digit, add to tens digit
    if(result != [] and result[0][2] > 0.40):
        return int(result[0][1]) + tens

    # inconclusive result (lots of white pixels but no clear form)
    return -1

# computes a similarity score over the window with the kernel
def img_detect(frame, window, kernel, total_filter, threshold):
    """
    computes a similarity score over the window with the kernel

    frame: A*B*3 image matrix
    window: list of 4 coordinates, x1,x2,y1,y2
    kernel: reference image, applied with convolution to window
    total_filter: identically shaped to the kernel, populated with ones
    threshold: required value to identify a successful match
    """
    # crop image to contain only specified window
    img_window = frame[window[0]:window[1], window[2]:window[3]].astype("float")
    # compute raw convoltion between the window and the kernel, sum over colour axis to yield A*B matrix
    i1 = cv2.filter2D(src=img_window, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_CONC8ANT).sum(axis=2)
    # compute sum of the squares of the pixel values over the size of the kernel
    frame_tot = cv2.filter2D(src=np.square(img_window), ddepth=-1, kernel=total_filter, borderType=cv2.BORDER_CONC8ANT).sum(axis=2)
    
    # to compute similarity, subtract magnitude of the window segment from magnitude of the kernel convolution result
    # effectively, cosine similarity
    conv_map = frame_tot - 2*i1

    mini = conv_map.min()

    ## visualize for debug purposes
    # visual = (conv_map - mini) * 255 / (maxi-mini)
    # visual = visual.astype("uint8")


    loc = conv_map.argmin()
    # if threshold is met, return True along with the location of the argmin ("match")
    return mini < threshold, [int(loc/len(conv_map[0])), loc%len(conv_map[0])]

def read_from_video(vid, img_hud, filter_hud, img_equip, filter_equip):
    """
    iterates through video, detecting and tallying every decrementation of equip

    vid: cv2.VideoCapture object, for reading in every video frame
    img_hud: A*B*3 hud image to be used for detecting menu
    filter_hud: A*B corresponding filter for img_hud
    img_equip: C*D*3 equip image template to search for
    filter_equip: C*D corresponding filter for img_equip
    """

    # read in first frame and get resolution
    ret, frame = vid.read()
    res = len(frame) / 720
    
    # search using diagonal lines for equip, running check_bounds for each pixel visited
    # once a check_bounds call for a pixel returns true, halt
    leave = False
    for j in range(int(200*res), int(800*res)):
        for k in range(min(j-int(200*res)+1, int(255*res))-1, -1, -1):
            i = int(450*res) + k
            if(frame[i][j-k][0] < 95 and frame[i][j-k][1] < 75 and frame[i][j-k][2] > 175 and abs(int(frame[i][j-k][0]) - int(frame[i][j-k][1])) < (40 + min(15, max(0, frame[i][j-k][2] - 178)))):
                mps, w_len = check_bounds(frame, i, j-k)
                if(mps):
                    leave = True
                    j = j-k
            if(leave): break
        if(leave): break
    
    if(not leave):
        raise Exception("Failed to find mps")
    
    # define search windows (x1,x2,y1,y2) for all 3 relevant search types
    # additional buffer is added to every side
    w_len = int(w_len*2.5)
    hud_detect_window = [int(100*res), int(360*res), j+int(w_len*0.45), j+int(w_len*0.95)]
    menu_window = [int(50*res), int(670*res), int(160*res), int(1120*res)]
    equip_window = [i+int(w_len*0.05), int(720*res), j-int(w_len*0.05), j+int(w_len*0.95)]

    search_dia = round(w_len / 18)

    # need to store results over multiple frames to reduce impact of visual noise
    read_queue = deque([0,0,0,0,0,0,0,0])

    prev_menu_frame = 1000
    current_value = 0

    # overall running total of every time equip is decremented (used)
    total_equip = 0

    # store time and magnitude of most recent decrementation, in case it reverts (caused by noise)
    prev_reduction = [0, -1000, 0]

    # iterate through every frame
    while(True):
        ret, frame = vid.read()
        # break if frame reader has reached end of file
        if(frame is None):
            break
        
        # when not in or proximal to menu_mode, only read every 4th frame
        if(i % 4 != 0 and prev_menu_frame > 15):
            continue

        # if in menu_mode, hud will be present on screen
        if(img_detect(frame, hud_detect_window, img_hud, filter_hud, -80000000)[0]):
            prev_menu_frame = 0

            # given menu_mode, search for equip within bounds defined by menu_mode
            detected_equip, loc = img_detect(frame, menu_window, img_equip, filter_equip, -12000000)

            # if equip is present on screen
            if(detected_equip):
                # set current_value to 0, as equip has not been decremented (still present)
                current_value = 0

                # read the number associated with equip
                loc = [loc[0]+menu_window[0], loc[1]+menu_window[2]]
                equip_value = read_num(frame, loc, search_dia)

                if(equip_value != -1 and equip_value < 30):
                    # if value is reasonable, push and pop queue
                    read_queue.append(equip_value)
                    read_queue.popleft()
                    # if large equip detected soon after a decrementation, decrementation was noise
                    # so revert it
                    if(i-prev_reduction[1] < equip_value*150 and equip_value >= prev_reduction[2] - 1):
                        total_equip -= prev_reduction[0]
                        prev_reduction = [0, -1000, 0]
            # if not detected, menu empty
            # assume value was fully decremented, save amount of decrementation
            elif(min(read_queue) != 0):
                current_value = sorted(read_queue)[3]

        # otherwise, standard mode
        else:
            if(prev_menu_frame == 0):
                # if last frame was menu, lock in decrementation during menu (if any)
                total_equip += current_value
                prev_reduction = [current_value, i, 0]
                current_value = 0
                read_queue = deque([0,0,0,0,0,0,0,0])

            prev_menu_frame += 1
            # given standard_mode, search for equip within bounds defined by standard_mode
            detected_equip, loc = img_detect(frame, equip_window, img_equip, filter_equip, -12000000)

            # if equip is detected, read number and add to read_queue
            if(detected_equip):
                loc = [loc[0]+equip_window[0], loc[1]+equip_window[2]]
                equip_value = read_num(frame, loc, search_dia)
                if(equip_value != -1 and equip_value < 30):
                    read_queue.append(equip_value)
                    read_queue.popleft()
            else:
                read_queue.append(0)
                read_queue.popleft()

            
            # if menu recently opened, can cause discrepancy in readings
            # only check for decrementations if >15 frames since menu
            if(prev_menu_frame > 15):
                # check if current frame and 5, 6 frames prior agree
                if(read_queue[-1] == read_queue[-6] and read_queue[-1] == read_queue[-5]):
                    reading = read_queue[-1]
                    # if large equip detected soon after a decrementation, decrementation was noise
                    # so revert it
                    if(i-prev_reduction[1] < reading*150 and reading >= prev_reduction[2] - 1):
                        total_equip -= prev_reduction[0]
                        prev_reduction = [0, -1000, 0]
                    if(current_value > reading):
                        # if current readings are lower than stored value, decrementation occured
                        # add decrementation to total
                        total_equip += current_value -  reading
                        prev_reduction = [current_value - reading, i, reading]
                    elif(current_value + 2 < reading < current_value + 15):
                        # if value has incremented by more than 2, not counted (unnatural source)
                        # remove this increase from the total
                        total_equip -= reading - current_value
                        prev_reduction = [0, -1000, 0]
                
            if(read_queue[-1] == read_queue[-6] and read_queue[-1] == read_queue[-5]):
                current_value = read_queue[-1]
    
    return total_equip


if(__name__ == "__main__"):
        
    eve_num = '15'

    cdict = {"vpode": 0, "njogb": 1, "qimpfcn": 2, "nvfjoae": 3, "beinpa": 4, "nioaeti": 5, "bsro": 6, "mgieroq": 7, "bosrj": 8, "eopjtp": 9}

    teams = [["P045", "P013", "P045", "P017"],["P025", "P057", "P042", "P023"],
    ["P090", "P011", "P035", "P038"],["P032", "P054", "False", "P118"],
    ["P013", "P062", "P048", "P135"],["P044", "P055", "P061", "P074"],
    ["P004", "P064", "P022", "P047"],["P099", "P097", "P096", "P067"],
    ["P066", "P049", "P059", "P127"],["P033", "P051", "P084", "P053"]]

    pla_list = []
    for team in teams:
        pla_list.extend(team)

    # assign each player one of two visual huds
    hud_list = ["P022", "P032", "P059", "P084"]
    pla_huds = {}
    for pla in pla_list:
        if(pla in hud_list):
            pla_huds[pla] = 1
        else:
            pla_huds[pla] = 0

    # initialize reference image for template matching
    img = cv2.imread("Equip.JPG")
    img_equip = img[1:37, 4:36, :]

    # blur and downsample image to fit correct dimensions
    blur = cv2.GaussianBlur(img_equip,(5,5),1)
    y_inds = np.asarray(range(0, 36, 2))
    x_inds = np.asarray(range(0, 32, 2))
    img_equip = blur[y_inds[:, None], x_inds, :]

    # initialize helper kernel for summing over window equivalent to reference image
    filter_equip = np.ones((18,16))

    # actual reference image is circular, zero out pixels in corners
    for i in range(18):
        for j in range(16):
            if((abs(i-9)*2 + abs(j-8) > 16)):
                img_equip[i][j] = [0,0,0]
                filter_equip[i][j] = 0

    # initialize red and blue reference images, used for detecting specific regions on screen
    img_blue = cv2.imread("blue_hud.JPG")
    img_blue = img_blue[:, 15:]
    img_red = cv2.imread("red_hud.JPG")
    img_red = img_red[13:, 3:]

    # all players use one of two huds, store both and select correct one using dict pla_huds
    huds = [img_blue, img_red]

    # initialize helper kernel for summing over region equivalent to hud images
    filter_hud = np.ones((23,40))

    # language reader, used for detecting values of equip (numerical)
    reader = easyocr.Reader(['en'])

    pla_stats = {}
    # iterate through every player and call video reader function, uses supplied hud and equip images
    for pla in pla_list:
        vid = cv2.VideoCapture(str(pla) + eve_num + ".mp4") 
        pla_stats[pla] = read_from_video(vid, huds[pla_huds[pla]], filter_hud, img_equip, filter_equip)

    check = 0