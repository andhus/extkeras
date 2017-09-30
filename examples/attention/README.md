# Canonical Recurrent Attention Example
There are many cases where the standard-encoding approach for seq-to-seq learning are highly inefficient. When encoding a sequence into a “final-state” vector using an LSTM (or any RNN) it is required that all relevant pieces of information as well as their spatial relation to each other are encoded in this single vector. If the order in which things happen is important (which it should be since we called it a sequence) - in particular if the spatial structure of the output sequence at least to some extent resembles that of the input - your are usually far better off using attention.

In the example in this folder we apply recurrent attention to solve a problem that only requires learning alignment/filtering of input to produce the right output. The input sequence consists of mainly zeros with random one-hot-encoding vectors added for a random subset of the timesteps. The model’s target is to reconstruct this sequence without the “empty slots”, i.e. pick out the one-hot-encoding in the right order.

![alignment](https://user-images.githubusercontent.com/5502349/31042881-14f47180-a5b2-11e7-9925-3b6499648f5b.png)

This is an overview of the standard seq-to-seq approach:

![standard-seq-to-seq](https://user-images.githubusercontent.com/5502349/31042886-253041fa-a5b2-11e7-8f75-953563fd3515.png)

...and this the attention version:

![attention-seq-to-seq](https://user-images.githubusercontent.com/5502349/31042888-2ebd993e-a5b2-11e7-9e13-070ca8135369.png)


## Performance
