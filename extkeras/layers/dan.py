from __future__ import division, print_function


"""Dynamic Associative Networks (DANs)

Yesterday me my (fewerish) sleep I came up with a exciting idea. Probably
someone has already tried the same thing but I'm surprised I haven't heard more
about it. It should be trivial to implement so I'm just gonna go ahead and test
it.

In regular FFW NNets we feed an input through a predefined set stack of
transformations to get an output. We then compare the output with a target and
use backprop to adjust the weights of the stack of transformations.
I've always felt this is nothing evenly remotely close to intelligence or
"reasoning/problem solving".


With DANs the idea is to instead define a *dynamic (recurrent) system*. We use
the input to determine the initial state of the system then let the system "run
freely" until its state *converges*. We then use the converged state (or some
transformation from it) as output and compare it to the target. We then again
just use backprop to adjust the weights defining the dynamic system.


This way can can sort of "forget about depth" of the network - we let it run an
undefined number of transformations until it's "done" (i.e. converges).
A recurrent dynamic system should also be able to reuse "sub-modules/programs"
more several times during analysis of one input which should allow for more
efficient use of its weights.

The the question is how to implement this. I came up wih an idea that is
essentially a generalisation of MLP but where information does not only have to
flow in a single direction. Just like a DAG is a special case of a DG the MLP
would be a special case of a DANs.

Moreover my idea is to connect nodes in the DAN by "association keys" rather
explicit connections. I.e. the strength of the connection between node A -> B
depends on how similar A:s outgoing associative key is with B:s incoming
associative key is. I find this setup interesting for a number of reasons:
    1. it somehow enforces sparsity in the connections (more on this later)
    2. it makes the number of weights as well as the computational time for
        evaluating a single time step scale only *linearly* with the number of
        nodes in stead of quadratically in a regular fully connected networks

I just had to get this down into writing before starting to implement...

Plan:
(-) implement basic DAN layer by using keras Recurrent layer... (hide in single
    layer later)
(-) write basic tests of step function
(-) use loss weights that gradually promotes convergence
(-) train on minimal mock problem
(-) train on mnist
(-) ...
"""

