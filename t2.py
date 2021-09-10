import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def stitch(imgmark, N=4, savepath=''): 

    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        img = img.astype(np.uint8)
        imgs.append(img)
    "Start you code here"
    
    
    def zero_paddding(image):    

        
        height_image = image.shape[0] 
        width_image = image.shape[1]
        third_image = image.shape[2]
        
        result = np.zeros((image.shape[0]+1000, image.shape[1] + 1000, 3), dtype=np.uint8)
        
        start = 500 
        result[500:500+image.shape[0],500:500+image.shape[1]]  = image
        
        
        
        return result    
    
        
    def compute_sift(img1):
        
        img1 = img1.astype(np.uint8)
        sift = cv2.xfeatures2d.SIFT_create()
        
        
        kp_image1, desc_image1 = sift.detectAndCompute(img1, None)
        return kp_image1, desc_image1    

    
    def SSD(desc_image1, desc_image2):
        list1 = []
        for i in range(len(desc_image1)):
            list2 = []
            for j in range(len(desc_image2)):
                error = (desc_image1[i] - desc_image2[j])
                euclid_distace = np.matmul(error.T, error)
                list2.append([i, j, euclid_distace])
#https://www.geeksforgeeks.org/python-sort-list-according-second-element-sublist/    
            list2.sort(key=lambda x: x[2])
            min_1 = list2[0]
            min_2 = list2[1]
    
            ratio = min_1[2]/min_2[2]
            if ratio < 0.75:
                list1.append(list2[0])
    
        return list1    
    
    def keypoint_value(kp_image1, kp_image2, value_set): 
        kp1 = []
        kp2 = []
        for i in range(len(value_set)):
            img1_keypoint = value_set[i][0]
            img2_keypoint = value_set[i][1]
            kp1.append(kp_image1[img1_keypoint])
            kp2.append(kp_image2[img2_keypoint])
            
        pointsA = []
        pointsB = []
        
        for eachPt in kp1:
            A = np.float32(eachPt.pt)
            pointsA.append(A)
        
        for eachPt in kp2:
            B = np.float32(eachPt.pt)
            pointsB.append(B)
        
        pointsA = np.array(pointsA)
        pointsB = np.array(pointsB)
        
        return pointsA, pointsB

    def compute_homography(pointsA, pointsB):
        Homography, mask = cv2.findHomography(pointsA, pointsB, cv2.RANSAC, 4.0)
        return Homography
    
    def do_warp(Homography, img1, img2):

        result = cv2.warpPerspective(img1, Homography, ((img1.shape[1]+img2.shape[1]), (img1.shape[0]+img2.shape[0])))
        return result    
    
    def reshape(result, img2):
        temp = np.zeros((result.shape))
        temp[:img2.shape[0], :img2.shape[1]] = img2
        temp = temp.astype(np.float32)
        result = result.astype(np.float32)
    
        return result, temp
    
    def stitch_blend(result,temp,alpha):
        alpha = 0.5
        beta = 1-alpha
        blended = beta*result + alpha*temp
        return blended       
    
    def get_descriptors(imgs):
        

        count_image = len(imgs)
        desc = []
        for i in range(count_image):
            key_points, descriptors =  compute_sift(imgs[i])
            desc.append(descriptors)
            
        return desc
    def get_overlap_matrix(list_images):
        
        no_of_images = len(list_images)
        match_val_mat = np.zeros((no_of_images,no_of_images))
        overlap_mat = np.zeros((no_of_images,no_of_images))
        for i in range(no_of_images):
            for j in range(no_of_images):
                value_set = SSD(list_images[i], list_images[j])
                no_of_matches = len(value_set)
                total_descriptors = len(list_images[i])
                if no_of_matches/total_descriptors >= 0.20:
                    match_val_mat[i][j]= no_of_matches
                    overlap_mat[i][j]= 1
        
        return match_val_mat, overlap_mat
    
    def get_sequence(overlap_mat,list_images, match_val_mat):
        check_max=[]
        for row in overlap_mat:
            val=sum(row)
            check_max.append(val)
        index_max = np.argmax(check_max)
        center_image = list_images[index_max]
        image_count = match_val_mat[index_max]
        
        sort_index = np.argsort(image_count)
        
        sort_index-=max(sort_index)
        
        sort_index=np.absolute(sort_index)
        
        
        image_seq_list=[list_images[sort] for sort in sort_index]
        
        image_seq_list=[(image) for image in image_seq_list]
        
        return image_seq_list

    def stitch_sequence(imgA, imgB):
        
        img1 = imgB.astype(np.uint8)
        img2 = imgA.astype(np.uint8)
        
        kp_image1, desc_image1 = compute_sift(img1)
        kp_image2, desc_image2 = compute_sift(img2)
        
        value_set = SSD(desc_image1, desc_image2)
        
        pointsA, pointsB = keypoint_value(kp_image1, kp_image2, value_set)
        
        Homography = compute_homography(pointsA, pointsB)
        
        result = do_warp(Homography, img1, img2)
        
        result, temp = reshape(result, img2)
        
        
        Final = np.zeros((result.shape))
        

        for i in range(Final.shape[0]):
            for j in range(Final.shape[1]):
                for k in range(Final.shape[2]):
                    if (result[i][j][k]) == 0 and (temp[i][j][k] ==0):
                        Final[i][j][k] = 0
                    else:
                        if temp[i][j][k] - result[i][j][k] >= 0:
                            Final[i][j][k] = temp[i][j][k] 
                        else:
                            Final[i][j][k] = result[i][j][k]
    
        return Final
    
    list_images = get_descriptors(imgs)
    
    
    match_val_mat, overlap_mat = get_overlap_matrix(list_images)
    print(overlap_mat)
    
    image_seq_list = get_sequence(overlap_mat,imgs, match_val_mat)

    new_img_list = []
    for i in image_seq_list:
        new_img_list.append(zero_paddding(i))

    input_img = new_img_list[0]
    for i in range(len(new_img_list)-1):
        img = stitch_sequence(input_img,new_img_list[i+1])
        
        input_img = img
    
    x = np.where(input_img != 0)
    x1 = np.min(x[0])
    x2 = np.min(x[1])
    x3 = np.max(x[0])
    x4 = np.max(x[1])
    
    # if width == 0 or height ==0:
    #     continue
    final_img = input_img[x1:x3, x2:x4]
    
    cv2.imwrite(savepath, final_img)
    
    return overlap_mat




if __name__ == "__main__":
#    task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
   # bonus
    overlap_arr2 = stitch('t3',N=4,  savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
