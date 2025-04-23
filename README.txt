
- agregated_sea_consumption_v9.py Ninth instance of the code
acts as a library or the functions developed on the previous version 
"sea-consumption-grid.py" is to call this pakage.
this is a streamlined version to compute the total of sea resources consumed given some input parameters
that where taken as a constant in the previous verions

imput parameters:
land_productivity or max land capacity
number_of_consumers
confg (configuration file containing 3 classes: the variables, parameters and limits for the simulation)

output:
norm_burned_land, norm_burned_sea, movements

is run by being called by sea-consumption-grid.py as
import agregated_sea_consumption_v9 as mc

- sea-consumption-grid.py expansion of the general code to compute under which circumstances there is sea consumption

the code runs trought a set range of parameter spaces and computes the amount of sea resources consumed after X time steps
returns an hisotgram of searesource consumed when varing several combinations of parameters
start with land-production vs number of consumers.

is run by > python sea-consumption-grid.py name_plot



