import numpy as np 
import os
from tensorflow.keras.utils import to_categorical
import skimage.io as dt
from skimage import transform 
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
import random
import cv2


SIZE=224 #224 recomended
rootdir='frame-dataset' #directory to your database, the directory contains 4 folders labelled as 4 classes (Handwritten, Codewalk, Misc, Slides)
test_img_per_class=240 # represents no of test sample u want to keep per class

''' Checking database for imbalance, returns the maximum number of imaged can be used to create balance dataset '''

def create_dataset_report(root_dir):
    classes = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]
    total=[]
    for Class in classes:
        num = len(os.listdir(Class))
        print('No of images in '+ Class[14:] + ' is: '+str(num))
        total.append(num)
    return min(total)

max_img_num = create_dataset_report(rootdir)



''' Process dataset for feeding the network, returns images and labels list
 Label Encoding and One hot encoding of labels and Train Test split is done below'''

def process_dataset(root_dir,start_index,end_index):
    count = 0
    classes = [os.path.join(root_dir,f) for f in sorted(os.listdir(root_dir))]
    images = np.zeros(((end_index-start_index)*len(classes),SIZE,SIZE,3))
    labels = []
    for Class in classes:
        imagefiles = [os.path.join(Class,f) for f in os.listdir(Class)]
        for image in tqdm(sorted(imagefiles[start_index:end_index]), position=0, leave=True):
            if image.endswith(".png"):
                img = cv2.imread(image).astype('float32')
                img = cv2.resize(img,(SIZE,SIZE))/255
                images[count] = img
                labels.append(Class[14:])
            count += 1
    le = LabelEncoder()
    labels = le.fit_transform(labels) #labels are encoded as integers
    mapping = dict(zip(le.classes_, range(len(le.classes_)))) #prints which class is assigned to what label
    print(mapping)
    labels_one_hot = to_categorical(labels)
    
    # images, labels = shuffle(images, labels, random_state=0)
    return images,labels_one_hot


train_images,train_labels = process_dataset(rootdir,start_index=0,end_index=max_img_num-test_img_per_class)

os.mkdir('Train_Data')
np.save('Train_Data/images_train.npy',train_images)
np.save('Train_Data/labels_train.npy',train_labels)
del  train_images #depending on database it might be big enough to create OOM error, deleting for being on safeside.
del train_labels
print('Train Data successfully saved at Train_Date directory...')

test_images,test_labels = process_dataset(rootdir,start_index=max_img_num-test_img_per_class,end_index=max_img_num)
os.mkdir('Test_Data')
np.save('Test_Data/images_test.npy',test_images)
np.save('Test_Data/labels_test.npy',test_labels)
del test_images
del test_labels
print('Test_Data successfully saved at Test_Data directory...')



