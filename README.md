**********************************
Author:   Evan Dietrich
Course:   Comp 131 - Intro AI
Prof:     Santini

Assign:   Artificial Neural Networks
Date:     12/16/2020
File:     README.md
**********************************

Overview:
This program implements a multi-layer neural network to classify 3 types of
Iris plant. First, the program trains the ANN, and then classifies the plants
based on user input/data from 'data.txt'. 

Technical Breakdown:
gardens.py - Runs the program.  
README.md  - Instructions to run program and assignment overview.

Running the Program:
To run the program, ensure all downloaded files are on your machine, and run
"python3 gardens.py" on your terminal screen. This will start the program for
an automated running procedure, first training the neural network (after
splitting the data provided into train/validate/test), and then classifiying
the plants based on that input. The procedure given is controlled by the global
variables (essentially, parameter options for runtime) found in the 'IMPORTS +
GLOBALS' section of the 'gardens.py' file. The example run is with:

FILE = 'data.txt'
LEARN_RATE = 0.12
NUM_EPOCHS = 300

By editing the globals, you can test the Neural Network on different epochs &
learning rates to judge the model's performance under various conditions.
This allows the user to test our model's ability on the iris dataset with
a greater understanding of what the optimal conditions may be on this specific
dataset. Modifiying the LEARN_RATE parameter will allow the user to alter the
model's weights of the network, with respect to the loss gradient, essentially
altering the speed at which our model learns. Changing the NUM_EPOCHS parameter,
meanwhile, will adjust the number of times the model will pass through the full
dataset. On each epoch, the model will have the ability to update internal
model parameters.

While this dataset was fairly straightforward, we were asked to implement a 
multi-layer network, and so this has been done with the use of hidden layers
that retain the connection weight of the prior layers. In terms of architecture,
I made use of the back-propagation algorithm as referenced from class,
including functions for forward and backward propagation resepectively, as this
was a supervised learning project and it is a proven training method. I was not
greatly concerned by time/space complexity of this approach, as the dataset
was not huge.

When you run the program, you will notice the training accuracy and validation
accuracy scores (on every 30 epochs), as the number of epochs increases to 300.
When running with LEARN_RATE = 0.12, training and validation accuracy measures
converge at around .97 and 1.0, respectively, by approximately 150 epochs. In
the test set, accuracy was 1.0 at the end of the 300 epochs. On 150 epochs,
test accuracy came to 1.0 as well. It is worth noting, that this is not a set
certainty, in that the random split of train/valid/test will determine in some
sense how well the model performs in terms of accuracy, based on different
program runs & statistical randomness. However, performance should still be
fairly strong during the run regardless. If (by statistical rarity),
the model underperforms on your initial run, another run should suffice to show
its true, more regular, performance ability.

In modifying the learning rate, larger variance surfaced. I modified the
LEARN_RATE hyperparamter to .2 and noticed that the training accuracy was much
less stable, and while validation set accuracy came to 1.0 at the end of 300
epochs, testing accuracy came to only .95 or so. In thinking further about this
assignment, it would be interesting to perform a 'grid search' methodology to
tune these quasi-hyper-parameters to optimum conditions, although for the
purpose of this assignment, it did not seem as necessary as train/test accuracy
was very high regardless, and the data was not too noisy, nor incomplete.

Collaborators:
I completed this assignment with assistance from the student posts and
instructor answers on the Comp131 Piazza page, with the lecture material
(specifically the slides on forward and backward propagation functions),
and our class textbook. Further, I used sklearn libraries to succinctly prepare
the 'data.txt' contents into NN-approachable content (in terms of train/test),
as well as Pandas for initial file read-in, and NumPy for some math-based
functionality. For insight into model optimization methodology, especially in
regards to forward-and-backward propagation of the multi-layered-network, I
found an online article that provided math based understanding: "https://
towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd".
Lastly, I have some prior completed coursework from Tufts' Comp135 (Intro
Machine Learning) course, that used a similar dataset to the iris
classification problem we had here, as well as coursework with neural network
approaches that I consulted with & drew some inspiration from too.

Notes:
Testing my solution by altering the number of epochs did not seem to greatly
affect test accuracy beyond 150 epochs, which I believe makes sense given the
limited scope of the dataset (150 total data-points, with only 4 features). I
was intrigued, however, to how the learning rate affected accuracy/performance,
and would be intrigued to see how (on more noisy, real-world, data) altering
this parameter would affect our classification performance.