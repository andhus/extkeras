# Canonical Recurrent Attention Example

> “The notion of attention [...] is something that’s becoming a big thing in Machine Learning, you don’t want to look at all of the data all of the time, you want to be able to focus in on a small part. The neural network should be free to learn how to do that.” [Alex Graves, Mar 2015](https://www.youtube.com/watch?v=-yX1SYeDHbg)

There are many cases where the standard encoding-decoding approach for seq-to-seq learning is highly inefficient. When encoding a sequence into a “final-state vector" using an LSTM (or any RNN), it is required that all relevant pieces of information, as well as their spatial relation to each other, are encoded in this single vector. If the order in which things happen is important (which it should be since we called it a sequence) - and in particular if the spatial structure of the output sequence to some extent resembles that of the input sequence - you are usually better off using Attention Mechanisms.

In the code example in this folder, we apply recurrent attention to solve a problem that only requires learning of “alignment” or “filtering” of an input sequence to produce the right output sequence. The input consists of mainly zeros with (random) one-hot-encoding vectors added for a (random) subset of the time steps. The model’s target is to reconstruct this sequence without the “empty slots”, i.e. pick out the one-hot-encodings in the right order.

![alignment](https://user-images.githubusercontent.com/5502349/31042881-14f47180-a5b2-11e7-9925-3b6499648f5b.png)

This is an overview of the standard _Encoder-Decoder_ approach:

![standard-seq-to-seq](https://user-images.githubusercontent.com/5502349/31042886-253041fa-a5b2-11e7-8f75-953563fd3515.png)

...and this is the _Attentive-LSTM_ version:

![attention-seq-to-seq](https://user-images.githubusercontent.com/5502349/31042888-2ebd993e-a5b2-11e7-9e13-070ca8135369.png)


## Performance
Below is a benchmark of the standard _Encoder-Decoder_ approach vs the _Attentive-LSTM_ for the problem described, with input sequence length 30 and output sequence length 10 (i.e. there are 10 non zero columns that should be picked out). In both cases, a single standard LSTM layer with 32 units is used. The _Attentive-LSTM_ clearly outperforms the _Encoder-Decoder_. This is despite the fact that the _Encoder-Decoder_ has about twice as many weights (due to that one LSTM is used for encoding and a separate LSTM for decoding and very few weights are needed for the attention mechanism).
![losses_32units_len10-30](https://user-images.githubusercontent.com/5502349/31045863-34bcb8a2-a5ed-11e7-8cda-784b711c368c.png)

In the case above, both models seem learn the task even if the Attentive-LSTM learns much faster. Hower if we increase the length of input to 60 and output to 20 the difference in performance is more drastic and the _Encode-Decoder_ has a hard time solving the problem at all.
![losses_32units_len20-60](https://user-images.githubusercontent.com/5502349/31045865-413c7b9e-a5ed-11e7-9e88-5a293fd4f9a8.png)


Some more background here: [Stockholm-AI Reading Group - Session #2: Attention! ...and it’s use in Deep Learning](https://github.com/andhus/stockholm-ai-reading-group/blob/master/Session-%232-2017-08-23/Attention%20and%20it's%20use%20in%20Deep%20Learning.pdf)

Thanks for the attention...