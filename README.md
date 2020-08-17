# Image-Classification
This program determines if your image is an airplane, car, bird, cat, deer, frog, horse, ship, or truck. I used CIFAR-10 to train the model which is included in the Keras module. First I had to convert the pixel values of the dataset to a float and then normalize it by dividing by 255. Next, I changed them to categorical types instead of continuous.  
# Model
I created a sequential model. The first layer had 32 filters. Next I included a dropout to prevent overfitting. Then I had another layer with 32 filters again. Finally I had another layer with 512 neurons with a dropout following that. I then ran the model through SGD to optimize it.  
# Training
I trained two models, one with 10 epochs and another with 25 epochs.  

I also created a GUI to be able to run this ML code.  

![image](https://user-images.githubusercontent.com/54549208/90417888-f720ad80-e079-11ea-8bd3-030911ca3b7f.png)
![image](https://user-images.githubusercontent.com/54549208/90417925-09025080-e07a-11ea-96d2-e6b8385c6efd.png)
![image](https://user-images.githubusercontent.com/54549208/90417962-17e90300-e07a-11ea-8411-58169a38fc22.png)
