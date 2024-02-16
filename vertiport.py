'''
when ATC creates vetiports, create instances of vertiports at those locations, 
ATC also knows how many uavs I have in my system, 
UAVs are assigned to vertiports, 
Vertiports keep count of UAVs at its disposal 
Vertiport releases UAV at some random time only if it has UAVs 
A UAV is added to the Vertiport, when it arrives 
same UAV can be released with new end point from given Vertiport 
'''

'''The simulator - 

    maintains time, 
    Time starts at 0, continues till end(defined by user)
    
    How long should a simulation run, what determines the end of simulation


    Vanilla simulator runs - for a specified time defined by the user. 
                           - Understand what is steps, 


    FOR Training the RL algorithm - the simulation should run until the Auto_UAV reaches its endpoint
                                  - so simulation ends when Auto_UAV reaches its endpoint 
                                  - We run N number of simulations 
    
    '''


'''
every class is one object - every function does only one thing for that object 

UAV - needs -  start, end, 

Airspace  - needs - location name

ATC - needs to know - location of all UAV, location of all vertiport 

Vertiport - needs - location of vertiport, no of uav at station 

'''