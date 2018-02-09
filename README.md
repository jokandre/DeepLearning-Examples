# Mix of Examples for studying purpose

- Based on DeepSchool.io

### Goals
1. Make Deep Learning easier (minimal code).
2. Minimise required mathematics.
3. Make it practical (runs on laptops).
4. Open Source Deep Learning Learning.

### Installation
1. Install Docker https://www.docker.com/
2. Use the following commands to run from docker<sup>[1](#myfootnote1)</sup>.
```
git clone https://github.com/sachinruk/deepschool.io.git
cd deepschool.io
docker-compose up --build
```
3. Now go to `localhost:9000` on your browser to start using the jupyter notebooks.
4. (Optional) If you are on a mac/windows some of the examples may not work because the docker image may run out of memory. Hence under preferences in docker there is the option to increase the allocated memory. I have set it to 8GB. Run `docker-compose up` again if you reset memory.

See [here](./misc/windows_instructions.md) for installing on windows.

### Contents
0. Lesson 0: Introduction to regression.
1. Lesson 1: Penalising weights to fit better (scikit learn intro)
#### Mathematics (optional)
2. Lesson 2: Gradient Descent. Using basic optimisation methods.
3. Lesson 3: Tensorflow intro: zero layer hidden networks (i.e. normal regression).
4. Lesson 4: Tensorflow hidden layer introduction.
#### Deep Learning
5. Lesson 5: Using Keras to simplify multi layer neural nets.
6. Lesson 6: Embeddings to deal with categorical data. (Keras)
7. Lesson 7: Word2Vec. Embeddings to visualise words. (Tensorflow)
8. Lesson 8: Application - Bike Sharing predictions
9. Lesson 9: Choosing Number of Layers and more
10. Lesson 10: XGBoost - A quick detour from Deep Learning
11. Lesson 11: Convolutional Neural Nets (MNIST dataset)
12. Lesson 12: CNNs and BatchNormalisation (CIFAR10 dataset)
13. Lesson 13: Transfer Learning (Dogs vs Cats dataset)
#### Advanced Topics
14. Lesson 14: LSTMs - Sentiment analysis.
15. Lesson 15: LSTMs - Shakespeare.
16. Lesson 16: LSTMs - Trump Tweets.
17. Lesson 17: Trump - Stacking and Stateful LSTMs.
18. Lesson 18: Fake News Classifier


## YouTube playlist
Find the corresponding video tutorial here (not all notebooks have an associated video)
https://www.youtube.com/playlist?list=PLIx9QCwIhuRS1SPS9LHF7VjvZyM1g2Swz

