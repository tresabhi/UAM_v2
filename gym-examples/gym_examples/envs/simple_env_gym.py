'''
This is a recreation of CADRL env, in this recreation I will not be using any global CONFIG file,
rather will pass arguments to the env, and initialize the env, this will improve readability, 
and easy to make different kind of envs for training and testing(evaluating)

This env will have different kinds of non-learning agents each with their unique control policies(by unique I mean only two kind - for now).
Some properties of this environment, 

1) area of environment scales with number of agents - to maintain some sense of constant distance between agents at beginning 
2) will use sparse reward based on MIT-ACL 



Non_learning agent - 
1) 


Learning agent - 
1)




env desc - 
1) start-end location of agents 
2) non-learning agent control policy 
3) render 
4) GYM env super-class 
5) GYM API 

'''