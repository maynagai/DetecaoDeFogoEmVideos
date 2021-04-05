import cv2
#mport imutils

#PUT ALL VIDEOS WITH THE SAME NAME WITH A NUMBER IN THE END STARTING FROM 1
#example of videos on the folder {fire1.avi, fire2.avi, fire3.avi...}
file = "fire"
extension = ".avi"
#indicate where is the folder of videos
folder = "D:\\Codigos_Phyton\\videoProcess\\mivia_fire\\" #"D:\\TrabDoc\\datasets\\video_dataset\\fogo\\"
# the min area to draw a bounding box
min_area = 300

video_width = 400

#folders to store the frames resulting from background subtraction
folder2 = "frames_MOG/"
#folder1 = "frames_1stFrame/"

numVideos=29

numVideos=numVideos+1

for i in range(1,numVideos):
    cont =0;

    #Capture each video from the folder
    videoFile = file + str(i)
    camera = cv2.VideoCapture(folder + videoFile + extension)

    #Create the enviroment to do background subtraction

    #1 Mixture of Gaussian MOG -> createBackgroundSubtractorMOG()
    # http://www.ee.surrey.ac.uk/CVSSP/Publications/papers/KaewTraKulPong-AVBS01.pdf
    #   history- Length of the history. (200)
    #   nmixtures - Number of Gaussian mixtures. (5)
    #   backgroundRatio - Background ratio. (0.7)
    #   noiseSigma - Noise strength(standard deviation of the brightness or each color channel).0 means some automatic value. (0)
    #maskMOG = cv2.bgsegm.createBackgroundSubtractorMOG(history=3,nmixtures=4, backgroundRatio=0.2)

    # 2 Mixture of Gaussian MOG 2 -> createBackgroundSubtractorMOG2()
    #  http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf
    #   history	-       Length of the history. (500)
    #   varThreshold -  Threshold on the squared Mahalanobis distance between the pixel and the model to decide whether a pixel is well described by the background model.
    #                   This parameter does not affect the background update. (16)
    #   detectShadows - If true, the algorithm will detect shadows and mark them.
    #                   It decreases the speed a bit, so if you do not need this feature, set the parameter to false. (True)
    #maskMOG2 = cv2.createBackgroundSubtractorMOG2(history=3,varThreshold=70,detectShadows=True)

    # 3 Probabilistic Foreground Subtraction -> Bayesian Estimation
    #  http://goldberg.berkeley.edu/pubs/acc-2012-visual-tracking-final.pdf
    #  Parameters:
    #     initializationFrames - number of frames used to initialize the background models (120)
    #     decisionThreshold - Threshold value, above which it is marked foreground, else background (0.8)

    estrut = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    maskGMG = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=5, decisionThreshold=0.6)

    #While there is frames to read
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if not grabbed:
            break
        frame = imutils.resize(frame, width=video_width)
        # Applies the background subtraction creating a new mask
        #fgmaskMOG = maskMOG.apply(frame)

        #fgmaskMOG2 = maskMOG2.apply(frame)

        fgmaskGMG = maskGMG.apply(frame)
        fgmaskGMG = cv2.morphologyEx(fgmaskGMG, cv2.MORPH_OPEN, estrut)
        # display the results
        diffImg = cv2.bitwise_and(frame, frame, mask=fgmaskGMG)

        cv2.imshow("Original Video", frame)
        #cv2.imshow("MOG 1 ", fgmaskMOG)
        #cv2.imshow("MOG 2 ", fgmaskMOG2)
        cv2.imshow("GMG ", fgmaskGMG)
        cv2.imshow("IMG ", diffImg)
        key = cv2.waitKey(1) & 0xFF

        cont = cont+1
        if(cont%5 ==0):
            cv2.imwrite("D:\\TrabDoc\\datasets\\video_dataset\\res\\"+videoFile+"_frame_"+str(cont)+".jpg", diffImg)

        # save the frame
        # cv2.imwrite("frame%d.jpg" % count, frameDelta)
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()






