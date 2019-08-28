Part 1
------
- [x] Explore emnist dataset
- [x] Train a few models based on different networks to recognize characters
- [x] Generate lines dataset from emnist dataset
- [x] Build a new conv net model to detect characters in an image of fixed dimensions
- [x] Use CTC loss with LSTM on TimeDistrubted to imporve the predications on an image of fixed dimensions

- [ ] Explore IAM datasets
- [ ] Improve the model even more by:
    - [x] Wrap the LSTM in a Bidirectional() wrapper, which will have two LSTMs read the input forward and backward and concatenate the outputs
    - [x] Stack a few layers of LSTMs
    - [ ] Try recurrent dropout
    - [ ] Add BatchNormalization

Part 2
------
- [ ] Detect line regions in an image of a whole paragraph of text
    + given an image containing lines of text, returns a pixelwise labeling of that image, with each pixel belonging to either background, odd line of handwriting, or even line of handwriting
    - [ ] Explore IAM forms images
    - [ ] Build an FCN model to do this task. The crucial thing to understand is that because we are labeling odd and even lines differently, each predicted pixel must have the context of the entire image to correctly label -- otherwise, there is no way to know whether the pixel is on an odd or even line.
    
Part 3
------
- [ ] Serve predictions using a Flask web app
 
Extras
------
- [ ] Try U-Net architecture to solve the line detection model
- [ ] Try Dataturks for data labeling
- [ ] Try setting up CircleCI

 