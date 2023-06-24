Classifier

db_getter.py: Import images from the camera to train the model.

In 'classes', input the amount of classes you want to classify and press 'Enter'.

Now the program is on input mode.

If you press 'Enter' again the program raise a camera window, at this point the
program is taking photos from the camera and classifying them as class 0 to train
the model.

Press 'q' to exit the camera and finish to make the class data.

Repeat this process until finishing with the amount of classes.

training.py: Trains the selected model.

You can change the model architecture based on the application you want to give it.
You just have to uncomment the model architecture you want to build on and comment the other
model definition. (Default shufflenet)

app.py: Interactive program for real time classification.

