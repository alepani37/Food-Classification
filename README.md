<h1>FOOD CLASSIFICATION</h1>

THIS REPOSITORY IS A MODIFIED VERSION OF THE TUTORIAL https://github.com/lambertoballan/handsonbow.<br>

With this repository is possible to train a classifier that predict che class of 5 different food class using lbp and bag of visual words approach.<br>

In the img folder you can find the images of the dataset. This is a subset of the "food image classification dataset" https://www.kaggle.com/datasets/harishkumardatalab/food-image.<br>
In the matlab folder is possible to find all the file used for dataset creation, feature extraction, training e results visualization.<br>
In particulr in the following files is possible to do the following operations (note that dataset creation, training e results visualization is the same in all the following files. It changes the feature extraction method):<br>
-test_color_lbp: lbp for each color extraction.<br>
-test_color_sift: multiscale dense sift for each color channel extraction.<br>
-test_color_sift_lbp: the previous two methods combined. <br>
-test_lbp: lbp feature extraction.
-test_sift: sift feature extraction with the following sampling -> point sampling, dense sampling, multiscale dense sampling. The sampling method must be selected at the beginning.<br>
-test_sift_lbp: the previous two methods combined. <br>
-test_sift_pyramid_4: 4-quadrant pyramidal sampling using multiscale dense sampling sift.
