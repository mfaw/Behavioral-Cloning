import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from scipy.stats import bernoulli

##Class that includes the functions to preprocess and augment the data
class preprocessing :

    def __init__(self , Csvfile,imageDIR) -> None:
        self.Csvfile = Csvfile
        self.imagesDIR = imageDIR
        self.data = pd.read_csv(Csvfile , header = None)
        self.data = self.data.values
        self.showHistogram()
        
    ## draw the histogram of the data
    ## this shows the data distribution
    def showHistogram(self):
        n, _, _ = plt.hist(self.data[: , 6], 100, facecolor='green')
        plt.show()

    ################################################ Balance Data #####################################################
    ## Data collected is not balanced and this is shown in the histogram of the steering angle
    ## Histogram shows the distribution of the data
    def balance_data(self , data,N=60, K=1,  bins=100):
        '''
        To balance the data, we need to reduce the number of high bins - the bins include the data of the steering angle. 
        A histogram is plotted and the bins are sorted to find the largest K bins.
        The indices of the data in the largest bins are collected to be removed from the dataset.
        After this function, the number of data is reduced because images were removed completely. 
        
        angles = angles array
        K = maximum number of bins to be removed
        N = minimum number of images in a bin
        bins = number of equal-width bins in the range
        '''
        #get the steering angle from the data (csv file)
        angles = list(data[: , 6])
        
        # n = the value of histogram bins (the height of the bin) === nbins
        # bins = the edges of the bins === nbins + 1
        n, bins, _ = plt.hist(angles, bins=bins, color= 'orange', linewidth=0.1)
        angles = np.array(angles)
        n = np.array(n)
        
        ##sort the bins 
        idx = n.argsort()[-K:][::-1]    # find the largest K bins
        del_ind = []                    # collect the index which will be removed from the data
        ##loop over the largest K bins
        for i in range(K):
            ##check if the value of the ith largest bin is greater than N 
            if n[idx[i]] > N:
                #find the elements in the angles array that are between the bin's edge (within the bin's range)
                ind = np.where((bins[idx[i]]<=angles) & (angles<bins[idx[i]+1]))
                #flatten the array containing the elements 
                ind = np.ravel(ind)
                #shuffle the elements randomly
                np.random.shuffle(ind)
                #extend list "del_ind" to add elements leaving the last N elements behind
                del_ind.extend(ind[:len(ind)-N])

        counter = 0
        for i in del_ind:
            #remove the elements (removing the images from the data)
            data = np.delete(data, i - counter, 0)
            counter+=1

        ##return data after removing the data from it
        return data
    
    # remove the highest bin
    def clearBins(self , bins = 1):
        print(len(self.data))
        #call the balance data function to remove the highest bin
        self.data = self.balance_data(self.data , K = bins)
        print(len(self.data))
    
    ################################################ Flipping Images #####################################################
    def flip(self , img , angle):
        # randomly flip the image 
        flag = np.random.randint(0,2)
        if(flag == 1):
            # flip image vertically
            img = cv2.flip(img , 1)
            # multiply steering angle by -1 because image was flipped
            angle = -1 * angle
        return img , angle

    ################################################ Random Gamma Correction #####################################################
    # this function makes dark colors more vivid and light colors more light
    # it changes the difference between dark and light areas
    def random_gamma(self , image):
            gamma = np.random.uniform(0.4, 1.5)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table) 

    ################################################ Random Brightness #####################################################  
    # randomly change the brightness of the image   
    def random_brightness(self , image):
        #changing brightness in HSV space
        image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
        image1[:,:,2] = image1[:,:,2]*random_bright
        image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
        return image1

    ################################################ Cropping Images #####################################################
    # cropping the image
    def cropImage(self , image , angle , top_percent = 0.3, bottom_percent = 0.1):
        '''
        cropping the image from the top and bottom only 
        cropping is to remove irrelevant data from the image
        '''
        # top removes the sky 
        top = int(np.ceil(image.shape[0] * top_percent))
        # bottom removes the car 
        bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
        return image[top:bottom, :] , angle

    ################################################ Random Shear #####################################################
    #randomly shearing the image horizontally  
    def random_shear(self , image,steering,shear_range = 150):
        rows,cols,ch = image.shape
        dx = np.random.randint(-shear_range,shear_range+1)
        #    print('dx',dx)
        random_point = [cols/2+dx,rows/2]
        pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
        pts2 = np.float32([[0,rows],[cols,rows],random_point])
        # shearing angle
        dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0    
        M = cv2.getAffineTransform(pts1,pts2)
        image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
        # change steering angle proportionally to shearing angle
        steering +=dsteering
        
        return image,steering

    ################################################ Subsampling the dataset #####################################################
    def subsampling(self):
            '''
                take of some of the steering angles based on how they are frequent in the dataset
                the more the same steering angle occure the higher the probability of it being droped of 
                by using the markov sambling method which is typically was used in NLP problmes
            '''
            threshold = 1e-6
            a = list(self.data[ : , 6])
            angles_counts = Counter(a)
            total_count = len(a)
            freqs = {angle: count/total_count for angle, count in angles_counts.items()}
            p_drop = {angle: 1 - np.sqrt(threshold/freqs[angle]) for angle in angles_counts}
            index = 0
            delet_index = []
            for i in a:
                if np.random.random() >= p_drop[i]:
                    delet_index.append(index)
        #             data = np.delete(data, index, 0)
                index+=1
                
            print(len(delet_index) , len(self.data) , len(delet_index) / len(self.data) )
            counter = 0
            for i in delet_index:
                self.data = np.delete(self.data, i - counter, 0)
                counter+=1
            print(len(self.data[: , 6]))

    ################################################ Generator #####################################################
    # this is to load the data in memory
    def Generator(self , batchSize = 128):
        batch_x = []
        batch_y = []
        print(len(self.data))
        while(1):
            # randomly shuffle the data in the batch
            random_idx = np.random.randint(0 ,len(self.data) , batchSize);
            for index in random_idx:
                choice = np.random.choice([1 , 3 , 5])
                img = plt.imread(','.join(self.data[index][choice-1:choice+1]).strip())
                angle = np.float32(self.data[index][6]) 
                # angle correction - because angle collected is from center image so left and right steering angles must be corrected
                # choice 3 are the left images
                if choice == 3:
                    angle += 0.222
                # choice 5 are the right images
                elif choice == 5:
                    angle -= 0.222
                
                # Bernoulli distributed discrete random variable with probability of success 0.8 
                head = bernoulli.rvs(0.8)

                # randomly shear images if head is 1
                if head == 1:
                    img, angle = self.random_shear(img, angle)

                # preprocessing of images by cropping  
                img , angle = self.cropImage(img,angle,0.38 , 0.137)
    #             crop_2(img,0.35 , 0.1)

                #head = bernoulli.rvs(0.8)
                #if head == 1:
                #    img = self.random_brightness(img)

                # preprocessing of images by cropping  
                img , angle = self.flip(img , angle)
                # preprocessing of images by gamma correction  
                img = self.random_gamma(img)
                # resize image to be 64 x 64 after preprocessing 
                img = cv2.resize(img , (64 , 64))
                # changing color of images from RGB to HSV 
                img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
                
                # batch of images ready to be used by model for training
                batch_x.append(img)
                batch_y.append(angle)
            
            # generator function so we use yield
            yield np.array(batch_x), np.array(batch_y)
            batch_x = []
            batch_y = []
        