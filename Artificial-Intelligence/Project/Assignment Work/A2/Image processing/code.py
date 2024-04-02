import cv2

# get original image size
path_to_original_image = 'Original_Image/S055_005_00000044.png'
original_image = cv2.imread(path_to_original_image) # read the image
height, width = original_image.shape[:2] # get height and width

image_to_be_transformed = ['karthik/images/happy1.jpg', 'karthik/images/happy2.jpg', 'karthik/images/happy3.jpg', 'karthik/images/happy4.jpg', 'karthik/images/happy5.jpg', 'karthik/images/not6.jpg', 'karthik/images/not7.jpg', 'karthik/images/not8.jpg', 'karthik/images/not9.jpg', 'karthik/images/not10.jpg', 'karthik/images/not11.jpg', 'karthik/images/not12.jpg'] # list of path to images to be transformed
transformed_images = [] # list of test images that have been transformed
                                        # these are opencv image objects
for i in range(len(image_to_be_transformed)):
     temp_ = cv2.imread(image_to_be_transformed[i])
     temp_ = cv2.resize(temp_, (height, width))
     transformed_images.append(temp_)
     cv2.imwrite("karthik/"+image_to_be_transformed[i].split("/")[-1], temp_) # store the resized image for future use