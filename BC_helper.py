'''
Data augmentation help functions
Credit to kaskmann at
https://github.com/ksakmann/CarND-BehavioralCloning/
'''
import cv2
import numpy as np

def read_next_image(lcr,X_train,Y_train, X_left=None, X_right=None):
    # assume the side cameras are about 1.2 meters off the center and the offset to the left or right
    # should be be corrected over the next dist meters, calculate the change in steering control
    # using tan(alpha)=alpha
    offset=1.0
    dist=20.0
    steering = Y_train
    if lcr == 0:
        image = X_left
        dsteering = offset/dist * 360/( 2*np.pi) / 25.0
        steering += dsteering
    elif lcr == 1:
        image = X_train
    elif lcr == 2:
        image = X_right
        dsteering = -offset/dist * 360/( 2*np.pi)  / 25.0
        steering += dsteering
    else:
        print ('Invalid lcr value :',lcr )

    return image,steering

def random_crop(image, new_height, new_width, steering=0.0,tx_lower=-20,tx_upper=20,ty_lower=-2,ty_upper=2,rand=True):
    # we will randomly crop subsections of the image and use them as our data set.
    # also the input to the network will need to be cropped, but of course not randomly and centered.
    shape = image.shape
    col_start,col_end =abs(tx_lower),shape[1]-tx_upper
    horizon=60;
    bonnet=136
    if rand:
        tx= np.random.randint(tx_lower,tx_upper+1)
        ty= np.random.randint(ty_lower,ty_upper+1)
    else:
        tx,ty=0,0

    #    print('tx = ',tx,'ty = ',ty)
    random_crop = image[horizon+ty:bonnet+ty,col_start+tx:col_end+tx,:]
    image = cv2.resize(random_crop,(new_height, new_width),cv2.INTER_AREA)
    # the steering variable needs to be updated to counteract the shift
    if tx_lower != tx_upper:
        dsteering = -tx/(tx_upper-tx_lower)/3.0
    else:
        dsteering = 0
    steering += dsteering

    return image,steering

def random_shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering

    return image,steering

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_flip(image,steering):
    coin=np.random.randint(0,2)
    if coin==0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering


def generate_training_example(X_train, X_left, X_right, Y_train, new_height,
                              new_width):
#    print('training example m :',m)
    lcr = np.random.randint(0,3)
    #lcr = 1
#    print('left_center_right  :',lcr)
    image,steering = read_next_image(lcr, X_train, Y_train, X_left, X_right)
#    print('steering :',steering)
#    plt.imshow(image)
    image,steering = random_shear(image,steering,shear_range=100)
#    print('steering :',steering)
#    plt.figure()
#    plt.imshow(image)
    image,steering = random_crop(image, new_height, new_width, steering,
                                 tx_lower=-20,tx_upper=20,ty_lower=-10,
                                 ty_upper=10)
#    print('steering :',steering)
#    plt.figure()
#    plt.imshow(image)
    image,steering = random_flip(image,steering)
#    print('steering :',steering)
#    plt.figure()
#    plt.imshow(image)

    image = random_brightness(image)
#    plt.figure()
#    plt.imshow(image)

    return image,steering

def generate_validating_example(X_val, Y_val, new_height, new_width):
    x,y = read_next_image(1, X_val, Y_val)
    return random_crop(x, new_height, new_width, y, tx_lower=0, tx_upper=0,
                       ty_lower=0, ty_upper=0)
