![image](https://github.com/user-attachments/assets/70683034-17da-4cb2-acfc-a16a546a428c)

![image](https://github.com/user-attachments/assets/6541fb73-f3a1-422f-ab81-ee305ef51a6f)


main idea: 
define the transformation between color intensity cdfs of images as the "editing" 
approximate cdf as polynomial? maybe order 5? then can quantify transformation
(may not need to fit at all, maybe can take value in each bin as value and do some inner product or analogous to scalar quantify "closeness")
tweak parameters such as exposure, saturation, blur etc then compute cdf, optimizing for closeness to ideal cdf

usage idea:
have some presets for certain kinds of editing like night photos or nature etc.
for similar kinds of photos, start with an example or a preset then add some noise to the transformation to see what looks the best. basically providing data for retraining, then apply new transformstion to whole set of images to be edited.
