
import cv2
import numpy as np
import operator
WHITE = [255, 255, 255]
BLACK = [0, 0, 0]
RED = [0, 0, 255]
LOWERGREEN = 74
UPPERGREEN = 175
HEIGHTCOMPTHRESHHOLD = 8
LETTERCOMPOFFSET = 30
def myMedianBlur(image):
    height, width, channels = image.shape
    for channel in range(channels):
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                neighbours = []
                for iNeighbours in range(-1, 2):
                    for jNeighbours in range(-1, 2):
                        neighbours.append(image[i + iNeighbours][j + jNeighbours][channel])
                neighbours.sort(reverse=True)   #sort pixels and pick median as new pixel
                image[i][j][channel] = neighbours[4]
    return image
def RGBtoHSVConversion(pixel):
    rPrime = pixel[2]/255.0
    gPrime = pixel[1]/255.0
    bPrime = pixel[0]/255.0

    cmax = max(rPrime, gPrime, bPrime)
    cmin = min(rPrime, gPrime, bPrime)

    delta = cmax - cmin
    h = 0
    if delta:
        if cmax == rPrime:
            h = (((gPrime - bPrime)/delta) * 60 + 360)%360
        elif cmax == gPrime:
            h = (((bPrime - rPrime)/delta) * 60 + 120)%360
        elif cmax == bPrime:
            h = (((rPrime - gPrime)/delta) * 60 + 240)%360
    s = 0
    if cmax:
        s = delta/cmax
    v = cmax

    return (h, s, v)
def isGreen(image, i, j): #given a pair of indices checks if the pixel on that position has a green hue
    h,_,_ = RGBtoHSVConversion(image[i][j])
    if h > LOWERGREEN and h < UPPERGREEN:
        return True
    return False
#def hasHue(image, i, j):
#    h,_,_ = RGBtoHSVConversion(image[i][j])
#    return h != 0

def generateSeeds(image): #generates the segments for each character in the form of a dictionary
    height, width, _ = image.shape   #where the key is the id and the value is a list of index pairs
    segments = {}
    visitedMatrix = np.zeros([height, width])
    currentRegionId = 0
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if not visitedMatrix[i][j]:
                if isGreen(image, i, j): #if a pixel is green and unvisited it means a new character segment was found
                    visitedMatrix[i][j] = 1
                    segments.update(bfsTraversal(image, i, j, currentRegionId, visitedMatrix))
                    currentRegionId += 1
                else:                   #any pixel that is a background pixel will be turned black(including potential white ones)
                    image[i][j] = BLACK
    return segments, image



    #return queue

def bfsTraversal(image, i, j, id, visitedMatrix):
    segment = {id:[(i, j)]}
    queue = [(i, j)]

    while queue:
        (iCurr, jCurr) = queue.pop(0)
        image[iCurr, jCurr] = WHITE
        neighbours = [(iCurr + rowDiff, jCurr + colDiff) for rowDiff in range(-1, 2) for colDiff in range(-1, 2)]
        neighbours.remove((iCurr, jCurr))
        for neighbour in neighbours:
            if neighbour not in segment[id] and isGreen(image, neighbour[0], neighbour[1]): #add all non-visited green neighbours
                segment[id] += [neighbour]
                queue += [neighbour]
                visitedMatrix[neighbour[0], neighbour[1]] = 1
    #optional: print the last identified character. Useful as a loading timerand for debugging
    print("BFS Traversal reached character" + str(id + 1))
    return segment

def getSegmentBorders(image, segments): #creates a new dictionary containing key-value pairs
    segmentBorders = {}                 #consisting of the segment ids and tuples representing the minimum and maximum
    height, width, _ = image.shape      #points for each segment
    for segmentKey, segmentVals in segments.items():
        minCoords = [height, width]
        maxCoords = [-1, -1]
        for segmentVal in segmentVals:
            if segmentVal[0] < minCoords[0]:
                minCoords[0] = segmentVal[0]
            elif segmentVal[0] > maxCoords[0]:
                maxCoords[0] = segmentVal[0]
            if segmentVal[1] < minCoords[1]:
                minCoords[1] = segmentVal[1]
            elif segmentVal[1] > maxCoords[1]:
                maxCoords[1] = segmentVal[1]
        segmentBorders[segmentKey] = (minCoords, maxCoords)
    return segmentBorders

def borderLetters(image, segments):  #graphically outlines the borders as a red box
    segmentBorders = getSegmentBorders(image, segments)
    for segmentKey, (minCoords, maxCoords) in segmentBorders.items():
        for col in range(minCoords[1], maxCoords[1]):
            image[minCoords[0]][col] = RED
            image[maxCoords[0]][col] = RED
        for row in range(minCoords[0], maxCoords[0]):
            image[row][minCoords[1]] = RED
            image[row][maxCoords[1]] = RED
    return image, segmentBorders

def splitIntoWords(segmentBorders): #groups the segment ids as words
    words = []
    currWord = []
    nrSegments = len(segmentBorders)
    for i in range(nrSegments - 1):
        currWord += [i]
        (minCoords, maxCoords) = segmentBorders[i]
        (nextMinCoords, nextMaxCoords) = segmentBorders[i + 1]
        if (nextMinCoords[1] - maxCoords[1] > 30) or (nextMinCoords[0] - maxCoords[0] > 50): #if there's a space or endline
            words += [currWord]                                                            #then we add the current word group and
            currWord = []                                                                  #create a new one
    currWord += [nrSegments - 1]     #adds the last segment to the last word
    words += [currWord]
    return words
def compare(elem1, elem2): #auxiliary comparison method
    (minCoords1, maxCoords1) = elem1 #uses y coordinate as primary criteria and x coordinate as secondary criteria
    (minCoords2, maxCoords2) = elem2
    if abs(minCoords2[0] - minCoords1[0]) > HEIGHTCOMPTHRESHHOLD: #if the difference between the y coordinte is smaller than a threshhold
        return minCoords1[0] - maxCoords2[0]  #the segments are considered equal in height
    return minCoords1[1] - minCoords2[1]

def sortSegmentBorders(segmentBorders): #sorts segments in terms of their minimum coordinates,
    nrSegments = len(segmentBorders)    #first in terms of y coordinate and if it is equal in terms of x coordinate
    for i in range(0, nrSegments - 1):
        for j in range(nrSegments - 1):
            if (compare(segmentBorders[j], segmentBorders[j + 1]) > 0):
                temp = segmentBorders[j]
                segmentBorders[j] = segmentBorders[j + 1]
                segmentBorders[j + 1] = temp
    return segmentBorders
def compareToLetter(image, D, minCoord, templateImage, letter): #compares the pixels of a segment to a
    letterKey, letterCoords = letter                            #character in the template image with the same borders
    Y, X = D                                                    # + an error eliminating offset
    nrPixelsEqual = 0
    for i in range(Y):
        for j in range(X + LETTERCOMPOFFSET):
            if all(image[minCoord[0] + i][minCoord[1] + j] == templateImage[letterCoords[0] + i][letterCoords[1] + j]):
                nrPixelsEqual += 1
    return (letterKey, nrPixelsEqual / (Y * X)) #returns index of similarity as ratio between number of equal pixels
                                                # and border area
def findBestResemblance(segmentBorder, image, templateImage, template):
    (minCoords, maxCoords) = segmentBorder
    D = (maxCoords[0] - minCoords[0], maxCoords[1] - minCoords[1])
    # get index of similarity for all template characters
    resemblanceValues = list(map(lambda x: compareToLetter(image, D, minCoords, templateImage, x), template.items()))
    return max(resemblanceValues, key=operator.itemgetter(1))[0] #get the max similarity index and return it's corresponding character

def generateText(image, templateImage, segmentBorders, template, words): #main text Generation method
    currentWordIndex = 0
    letterCount = 0
    text = ''
    for segmentBorder in segmentBorders.values():
        text += findBestResemblance(segmentBorder, image, templateImage, template) #adds the character that was the best match
        letterCount += 1
        # if we reached the end of the word we will add a space separator then refresh the letter count and current word
        if letterCount == words[currentWordIndex] and currentWordIndex != len(words) - 1:
            text += ' '
            currentWordIndex += 1
            letterCount = 0
    return text




            # Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image = cv2.imread("input/input.png", 1);
    templateImage = cv2.imread('input/sablon.png', 1)
    with open('input/sablon.txt', 'r') as fin:
        lines = fin.readlines()

    template = {line.split()[0]:(int(line.split()[2]), int(line.split()[1])) for line in lines}
    #apply custom MedianBlur
    image = myMedianBlur(image)
    #generate the segments
    segments, image = generateSeeds(image)

    #get the borders for the segments
    image, segmentBorders = borderLetters(image, segments)
    #sort segments so they can be split into words
    segmentBorders = sortSegmentBorders(segmentBorders)
    #split the ids into words
    words = splitIntoWords(segmentBorders)
    #since we know the letters are in order we can instead make a list of word length
    words = list(map(lambda x: len(x), words))
    #print the generated text
    greenText = generateText(image, templateImage, segmentBorders, template, words)
    with open("output/output.txt", "w") as fout:
        fout.write(greenText)
    #optional: output the image to check the borders
    #cv2.imshow("output.jpg", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite("output/output.jpg", image)


