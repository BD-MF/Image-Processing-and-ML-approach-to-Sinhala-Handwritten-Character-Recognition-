from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, os.path
import pandas as pd
import csv


def chaincode(gray):
    count0 = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    count5 = 0
    count6 = 0
    count7 = 0

    img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    done = False

    img =cv2.bitwise_not(img)
    original = img
    img = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)[1]
    ############## Discover the first point#################################
    for i, row in enumerate(img):
        for j, value in enumerate(row):
            if value == 255:
                start_point = (i, j)
                #print(start_point, value)
                break
        else:
            continue
        break
    ###################Chain code algorithm ##########################
    directions = [0, 1, 2,
                  7,    3,
                  6, 5, 4]
    dir2idx = dict(zip(directions, range(len(directions))))

    change_j = [-1, 0, 1,  # x or columns
                -1,    1,
                -1, 0, 1]

    change_i = [-1, -1, -1,  # y or rows
                0,      0,
                1, 1, 1]

    border = []
    #chain = []
    chain = []
    curr_point = start_point
    for direction in directions:
        idx = dir2idx[direction]
        new_point = (start_point[0] + change_i[idx], start_point[1] + change_j[idx])
        if img[new_point] != 0:  # if is ROI
            border.append(new_point)
            chain.append(direction)
            curr_point = new_point
            break
    count = 0

    while curr_point != start_point:
        # figure direction to start search
        b_direction = (direction + 5) % 8
        dirs_1 = range(b_direction, 8)
        dirs_2 = range(0, b_direction)
        dirs = []
        dirs.extend(dirs_1)
        dirs.extend(dirs_2)
        for direction in dirs:
            idx = dir2idx[direction]
            new_point = (curr_point[0] + change_i[idx], curr_point[1] + change_j[idx])
            # print("new", new_point)
            try:
                if img[new_point] != 0:  # if is ROI
                    border.append(new_point)
                    chain.append(direction)
                    curr_point = new_point
                    break
            except:
                continue

        if count == 20000: break
        count += 1

    # Getting length of list
    length = len(chain)
#    print(chain)
    i = 0

    # Iterating using while loop
    while i < length:
  #      print(chain[i])
        if (chain[i] == 0):
            count0 = count0 + 1
        if (chain[i] == 1):
            count1 = count1 + 1
        if (chain[i] == 2):
            count2 = count2 + 1
        if (chain[i] == 3):
            count3 = count3 + 1
        if (chain[i] == 4):
            count4 = count4 + 1
        if (chain[i] == 5):
            count5 = count5 + 1
        if (chain[i] == 6):
            count6 = count6 + 1
        if (chain[i] == 7):
            count7 = count7 + 1
        i += 1
    sum = count0 + count1 + count2 + count3 + count4 + count5 + count6 + count7

    Avg0 = round((count0 / sum) * 100, 2)
    Avg1 = round((count1 / sum) * 100, 2)
    Avg2 = round((count2 / sum) * 100, 2)
    Avg3 = round((count3 / sum) * 100, 2)
    Avg4 = round((count4 / sum) * 100, 2)
    Avg5 = round((count5 / sum) * 100, 2)
    Avg6 = round((count6 / sum) * 100, 2)
    Avg7 = round((count7 / sum) * 100, 2)


    chain_output = [ Avg0 , Avg1 , Avg2 , Avg3, Avg4, Avg5, Avg6, Avg7]
    return chain_output

    sum = count0 + count1 + count2 + count3 + count4 + count5 + count6 + count7
    Sum_Avg = Avg0 + Avg1 + Avg2 + Avg3 + Avg4 + Avg5 + Avg6 + Avg7
    # print("sum", sum)
    # print("AVG SUm", Sum_Avg)

def distance_profile(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    height, width = thresh.shape[:2]
    tb = list()
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []

    for y in range(0, height):
        for x in range(0, width):
            cp = thresh[y, x]
            if cp == 255:
                count1 += 1
            else:
                break
        # print(count1)
        arr1.append(count1)
        count1 = 0
    for y in range(0, height):
        for x in range(width - 1, 0, -1):
            cp = thresh[y, x]
            if cp == 255:
                count2 += 1
            else:
                break
        arr2.append(count2)
        count2 = 0

    for x in range(0, width):
        for y in range(0, height):
            cp = thresh[y, x]
            if cp == 255:
                count3 += 1
            else:
                break
        arr3.append(count3)
        count3 = 0
    for x in range(0, width):
        for y in range(height - 1, 0, -1):
            cp = thresh[y, x]
            if cp == 255:
                count4 += 1
            else:
                break
        arr4.append(count4)
        count4 = 0


    distance_output = [arr1, arr2, arr3, arr4]
    #print(distance_output)
    return distance_output
    plt.subplot(332), plt.imshow(img)
    plt.subplot(323), plt.plot(arr1), plt.title('left to right')
    plt.subplot(324), plt.plot(arr2), plt.title('Right to left')
    plt.subplot(325), plt.plot(arr3), plt.title('top to bottom')
    plt.subplot(326), plt.plot(arr4), plt.title('bottom to top')

    plt.show()


def density_features(img):

    countW = 0
    countB = 0
    ratio = 0
    windowCount = 0
    pixelCount = 0
    resultArray = np.array([])

    # window size
    windowsize_rows = 127
    windowsize_columns = 127
    pixel = []
    ratioA = []

    for r in range(0, img.shape[0] - windowsize_rows +1 , windowsize_rows):
        for c in range(0, img.shape[1] - windowsize_columns +1 , windowsize_columns):
            window = img[r:r + windowsize_rows, c:c + windowsize_columns]
            windowCount = windowCount + 1
           # print("Zone:", windowCount)
            for i in range(window.shape[0]):
                for j in range(window.shape[1]):
                    if (window[i][j] == 0).all():
                        countB = countB + 1
                    if (window[i][j] == 255).all():
                        countW = countW + 1
                        ratio = countB / (128 * 128)
                        ratio = str(round(ratio, 4))
            pixel.append(countB)
            ratioA.append(ratio)
            countW = 0
            countB = 0
            ratio = 0

    n_white_pix = np.sum(img == 255)
#    print("Total white pixels", n_white_pix)
    n_black_pix = np.sum(img == 0)


    upperDivision = 0
    lowerDivision = 0
    leftDivision = 0
    rightDivision = 0
    verticalDiff = 0
    horizontalDiff = 0
    ratioA = list(map(float, ratioA))
    # upper division total pixels
    for u in range(0, 7):
        upperDivision = round(upperDivision + ratioA[u], 4)
#    print("upperDivision", upperDivision)

    # lower division total pixels
    for l in range(8, 15):
        lowerDivision = round(lowerDivision + ratioA[l], 4)
    # left division total density
    leftDivision = round(
        ratioA[0] + ratioA[1] + ratioA[4] + ratioA[5] + ratioA[8] + ratioA[9] + ratioA[12] + ratioA[13], 4)
    # left division total denisty
    rightDivision = round(
        ratioA[2] + ratioA[3] + ratioA[6] + ratioA[7] + ratioA[10] + ratioA[11] + ratioA[14] + ratioA[15], 4)
    # Vertical density diffrence
    verticalDiff = round(upperDivision - lowerDivision, 4)
   # Horizontal density diffrence
    horizontalDiff = round(leftDivision - rightDivision, 4)
    # get 8 zones
    eightZones = []
    zone1 = round(ratioA[0] + ratioA[1], 4)
    eightZones.append(zone1)
    zone2 = round(ratioA[2] + ratioA[3], 4)
    eightZones.append(zone2)
    zone3 = round(ratioA[4] + ratioA[5], 4)
    eightZones.append(zone3)
    zone4 = round(ratioA[6] + ratioA[7], 4)
    eightZones.append(zone4)
    zone5 = round(ratioA[8] + ratioA[9], 4)
    eightZones.append(zone5)
    zone6 = round(ratioA[10] + ratioA[11], 4)
    eightZones.append(zone6)
    zone7 = round(ratioA[12] + ratioA[13], 4)
    eightZones.append(zone7)
    zone8 = round(ratioA[14] + ratioA[15], 4)
    eightZones.append(zone8)

    # add density features to vector
    #print(ratioA)
    resultArray = eightZones + ratioA
    resultArray.append(verticalDiff)
    resultArray.append(horizontalDiff)

    return resultArray

def csv_file(chain_output, distance_output,resultArray):
    with open('NEW_DATA_FINALLxx.csv', mode='a') as csv_file:
        fieldnames = ['Ratio0', 'Ratio1', 'Ratio2', 'Ratio3', 'Ratio4', 'Ratio5', 'Ratio6', 'Ratio7' ,'LR','RL','TB','BT',
                      'Density' ,'group','label','num']

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        # writer.writeheader()
        writer.writerow(
            {'Ratio0': chain_output[0], 'Ratio1': chain_output[1], 'Ratio2': chain_output[2], 'Ratio3': chain_output[3], 'Ratio4': chain_output[4],
             'Ratio5': chain_output[5], 'Ratio6':chain_output[6], 'Ratio7': chain_output[7],'LR':distance_output[0],'RL':distance_output[1], 'TB':distance_output[2],'BT':distance_output[3],
             'Density': resultArray ,'group':'A','label':'0DC40DD2','num':'49'})

def main():
    # image path and valid extensions
    imageDir = "D:/FYP/dataset/DataSet5/Characters/Hi/"  # specify your path here
    image_path_list = []
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff" , ".bmp"]  # specify your valid extensions here
    valid_image_extensions = [item.lower() for item in valid_image_extensions]

    for file in os.listdir(imageDir):
        extension = os.path.splitext(file)[1]
        if extension.lower() not in valid_image_extensions:
            continue
        image_path_list.append(os.path.join(imageDir, file))

    for imagePath in image_path_list:
        image = cv2.imread(imagePath)
        img = cv2.imread(imagePath)
        if img is None:
            print("Error loading: " + imagePath)
            # end this loop iteration and move on to next image
            continue
        elif img is not None:
            cv2.imshow(imagePath, img)

            cv2.destroyAllWindows()
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # wait time in milliseconds
        # this is required to show the image
        # 0 = wait indefinitely
        # exit when escape key is pressed
        key = cv2.waitKey(0)
        if key == 27:  # escape
            break

        size = np.size(gray)
        skel = np.zeros(gray.shape,np.uint8)
        #ret,img = cv2.threshold(img,127,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        done = False

        height = width = 512
        dim = (height, width)
        img_new = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
        (thresh, img_new) = cv2.threshold(img_new, 128, 255, cv2.THRESH_BINARY)

        #method chaincode passing the image
        chain_output = chaincode(gray)

        distance_output = distance_profile(img)

        resultArray = density_features(img_new)

        csv_file(chain_output,distance_output,resultArray)

if __name__=='__main__':
    main()