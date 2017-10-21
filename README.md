# RL study

RL agents. Many of them are broken at any given time.

### Desiderata in a general agent framework

- For every agent, you should be able to pass in your conv function and 
    maybe some stuff about the value function or whatever that you want
- Shared code for computing the n-step rewards
- Shared code for making sure that the input and output types match the 
    types required by the environment.
- Code for saving useful stuff to Tensorboard, eg the episode rewards
