# Panaroma
Stitching multiple images into one panoramic photo

I first evaluated the 4 given images to compute the sequence. I calculate the sift features for 
all four images. The image sequencing is based on the number of matches. Additionally, 
ratio of matches to total features has to be greater than 20%. From the overlap matrix, I get 
the image that has maximum matches with the other images. I sort them according to the 
number of matches. I then get the index position for each image to finally get them in 
descending order of their number of matches. This gives me the sequence.
Panorama
After getting the sequence I pass the images in pairs to stitch them. This task involves 
computing the SSD (Here is set the ratio test value to 0.75) from the sift features, computing 
homograph, and warping and stitching. The resulting image is sent again to form a 
panorama. Also, since I increase the size of the matrix each time to increase the size of the 
background image, in the final step I also remove extra spaces in the panorama so that I get 
only the panorama cut out. Additionally, I also added padding to my images to make sure 
they don’t crop out of the frame.

## This project was a part of my coursework at the University at Buffalo for the course Computer Vision and Image Processing.
