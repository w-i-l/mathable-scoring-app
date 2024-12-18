<h1>Mathable scoring application</h1>
<h2>A computer vision approach in computing the score in Mathable game</h2>
<img src='./data/all_pieces_board.jpg' width="400px" height="400px">



<br>
<hr>
<h2>About it</h2>
<p>The current application purpose is to compute the score in a Mathable game.</p>

<p>The Mathable board is a 14x14 grid where the players place the pieces. The pieces are numbers between 0 and 90. The goal of the game is to create equations by arranging the pieces on the board in such a way that 3 pieces in a row (horizontally, vertically) form a valid equation. The player that creates the most points receive from the equations wins.</p>

<p>The code purpose is to solve 3 main tasks:</p>
<ol>
    <li>determine the piece position placed on the board</li>
    <li>predict the piece value</li>
    <li>compute the score for the current round</li>
</ol>

<p>The rules for this example have been restricted to only have 2 players and the number of rounds is fixed to 50.</p>



<br>
<hr>
<h2>How to use it</h2>

>**NOTE**: For this you will need to have <code>conda</code> installed. If you do not have it, you can install it by following the instructions from the <a href="https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html">official documentation</a>.

<p>For using the provided application you will need to install the dependecies by running the following code:</p>


```bash
conda create -n mathable python=3.10
conda activate mathable
conda install matplotlib numpy opencv scipy tqdm
pip install tf_keras
```

<p>For creating a Python virtual environment you can use the following <a href="https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/">guide</a>. Then install the dependencies using <code>pip</code>.</p>


<p>This will create a new environment called <code>mathable</code> and install the required packages, using the <code>Python3</code>.</p>

<p>After that we will want to install the <code>Tensorflow</code> package with the MPS capabilities for the <code>MacOS</code>. This will ensure a faster processing time. For this we will use the following code:</p>

```bash
conda install -c apple tensorflow-deps
pip install tensorflow-macos==2.15.0
pip install tensorflow-metal==1.1.0
```

For `Windows` you can use the following code:

```bash
pip install tensorflow
```

<p>For other platforms or distribution, please refer to <a href"https://www.tensorflow.org/install/pip">offical documentation</a>.</p>

<p>After the instalation is complete, you will need to create a folder in the <code>data</code> directory called <code>test</code>. Here you can put all of the input data, with the following structure:</p>

<ul>
    <li>all images describing a move should have the game number, folowing by their move index and should be a <code>.jpg</code> file</li>
    <li>the annotation file containing the move made in the above image should have the same name as the image, but with the <code>.txt</code> extension</li>
    <li>the file used for delimiting the turns between the players should be called by its game number followed by <code>turns.txt</code></li>
</ul>

<p>So the folder structure should look like this:</p>

```
data
│
└───test
│   │
│   └───1_01.jpg
│   │   1_01.txt
│   │   1_02.jpg
│   │   1_02.txt
│   │   ...
│   │   1_turns.txt
```

<p>This structure is hard to read and understand, so we will provide a script that will create the structure for you. The script is called <code>data_organizer.py</code> and can be found in the <code>utils</code> directory.</p>

<p>It will arrange the data in the <code>data/test</code> directory, in subfolders for each game, called <code>game_1</code>, <code>game_2</code>, etc.</p>


<p>After organizing the data, you can run the <code>main.py</code> script with the following code:</p>

```bash
python main.py
```

>**Note:** If you encounter the following error: <code>ImportError: attempted relative import with no known parent package</code>  or <code>ModuleNotFoundError: No module named</code> make sure to run the script from the <code>src</code> folder and if that does not work, try to run the script with the following code: 

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

<p>This will generate the predictions regarding the piece placed, its position and the game score for each round. The results will be saved in the <code>data/output</code> directory.</p>

<br>
<hr>
<h2>How it works</h2>
<p>The algorithm is split into 3 main parts:</p>
<ol>
    <li>we extract the board and the piece placed this round from the image</li>
    <li>we analyze the piece and its position and we predict the piece value</li>
    <li>we compute the score for the current round</li>
</ol>

<h3>1. Extracting the board and the piece</h3>
<p>For this step we first have to find the board in the image. Since the image have such a dimensions and a surrounding background we are forced to first find the board contour, then cut the board.</p>
<p>For this step we will apply first a <code>filtering mask</code> to the image, then we will apply a <code>edge detection</code> algorithm to find the board's contour.</p>

<p>The image will look like this:</p>
<img src="./readme_assests/original_image.png" width="300px" height="300px">
<img src="./readme_assests/filtered_image.png" width="300px" height="300px">
<img src="./readme_assests/preprocessed_image.png" width="300px" height="300px">
<img src="./readme_assests/edges.png" width="300px" height="300px">
<img src="./readme_assests/processed_edges.png" width="300px" height="300px">
<img src="./readme_assests/contours.png" width="300px" height="300px">
<img src="./readme_assests/cropped_board.png" width="300px" height="300px">

<p>Having a way to only look on the board will help us in the second part. We will have to extract the piece played this turn. My approach was to make a absolute difference between the current board and the previous board. This way we will have the piece played this turn as the brightness spot on the resulted image.</p>
<p>On the resulted image will apply a <code>Gaussian blur</code> and a <code>thresholding</code> algorithm to get the piece. Looking like this:</p>
<div>
    <img src="./readme_assests/difference_image.png" width="300px" height="300px">
    <img src="./readme_assests/diff_threshold.png" width="300px" height="300px">
</div>

<h3>2. Analyzing the piece and its position</h3>
<p>The piece will be analyzed in 2 steps:</p>
<ol>
    <li>we will extract the piece's position from the image</li>
    <li>we will predict the piece value</li>
</ol>

<p>For the first step we will consider the piece having a (105, 105) pixels size. We will extract the piece's position by sliding over the resulted difference board and compute the mean value of the pixels. The position will be the one with the highest mean value.</p>

<p>For the second step we will use a <code>Convolutional Neural Network</code> to predict the piece value. I will provide more information about this in the <code>tech specs</code> section.</p>

<h3>3. Computing the score</h3>
<p>This part is not a computer vision related task, but a simple math computation. We will have to compute the score for the current round. The score is computed by creating all the possible equations from the pieces on the board and summing the points for each equation. The points are computed by the following rules:</p>

<p>For each equation we will have to check the following:</p>
<ul>
    <li>if the equation is valid</li>
    <li>if the piece was placed on a constraint tile</li>
    <li>if the piece can score from multiple equations</li>
    <li>if the piece was placed on a multiplier tile</li>
</ul>

<p>Besides the 3rd rule, the other rules are simple to implement. The 3rd rule is a bit more complex, since we will have to check if the piece can score from multiple equations. This is done by checking if the piece is part of another equation. If it is, we will have to check if the piece can score from both equations. If it can, we will have to compute the score for both equations and sum them, followed by multiplying the final score with a multiplier, if the piece was placed on a multiplying tile.</p>

<br>
<hr>
<h2>Tech specs</h2>
<h3>About CNN</h3>
<p>Using a Convolutional Neural Network requires a lot of data. Since we do not have a lot of data, we will augment the data extracted from the train images provided.</p>

<ol>
    <li>First we will extract the pieces from the games provided as train data. Then we were provided 2 images containing all pieces, one in a grid placement and the other in a group manner. Those can be found in <code>data</code> folder, <code>all_pieces_board.jpg</code> and <code>all_pieces_togheter.jpg</code>.</li>
    <li>Then we will augment the data by adjusting the brightness, satuiration and making a random zoom in the image.</li>
</ol>

<p>The above will provide us with enough data to train the CNN. For the original data we have the following plot, denoting the distribution of the pieces. Since we integrate 2 pieces from the auxiliary boards provided, it means that not all pieces are present in the train data. This is why we will have to augment the data.</p>

<img max src="./readme_assests/data_distribution.png" max-height="400px">

<p>We can clearly observe that some pieces are more likely to appear in the moves. This is very important, since we will have to balance the data for the CNN. We will have to augment the data in such a way that the distribution of the pieces is uniform.</p>

<p>For this I created a method that based on the desired number of examples for a class and its current number of examples, it will augment the data in such a way that the distribution is uniform. The method can be found in the <code>cnn/cnn_datase</code> folder, <code>get_no_of_attributes_for_class</code>.</p>

<p>By augmented the data, we can ensure that even if out detection algorithm is not perfect, the CNN will be able to predict the piece value. Below are some example of augmentaion:</p>

<img src="./readme_assests/0_aug_1.png" width="120px" height="120px">
<img src="./readme_assests/0_aug_2.png" width="120px" height="120px">
<img src="./readme_assests/0_aug_3.png" width="120px" height="120px">
<img src="./readme_assests/0_aug_4.png" width="120px" height="120px">

<br>
<p>After augmenting the data, we can see that the distribution of the pieces is uniform:</p>
<img max src="./readme_assests/after_data_distribution.png" max-height="400px">

<p>For training this model, I used ~200 examples for each class, with a batch size of 128. The pieces were resized to (40, 40) from (105, 105) for saving space and computation time. The model was trained for 20 epochs, with a learning rate of 0.001 and a decay of 0.0001. The model can be found in the <code>models</code> folder, <code>cnn_model</code>. The result of training is 100% accuracy for testing and evaluation.</p>
