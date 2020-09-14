## UNet+++ code for experiment

Train.py is for training, it will store model in directory "model" for each epoch in order to test easily, so the "model" directory can be nearly 4GB after training.

Test.py is for testing, it test every model in "model" directory and print the accuracy. We use the accuracy data to draw line chart

 