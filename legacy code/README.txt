Folder for code related to the costal resources project

- resources_costal_plain.py fisrt isntance of the code, basic generation and consumption of resources

is run by >python resources_costal_plain_0.1.py 


- matrix_resources_costal_plain_0.2.py second instance of the code, extends the previous consumption and production of resources to a NxM matrix and plots the result in a dinamic movie showing the matrix as cells and plots a graph of all the sea resources consumed. 

is run by >python matrix_resources_costal_plain.py name_plot


- vector_resources_costal_plain.py Third instance of the code,  modifies the previous code to only deal with a vector of consumption, simplifying the sea/land divided consumption of the previous "matrix" version. It also uses a different rutine of choosing where the  consumer will jump, previously was random, now it selects the cells within a radius r around itself that have up to a 80% of the cell containing the maximum of resources, and jumps randomly to any of these cells. 

is run by >python vector_resources_costal_plain.py name_plot

- vector_multiple_consumers.py Forth instance of the code, it adds several consumers to the vector. The jumping strategy is designed to avoid that two consumers fall in the same cell. If they do, the cells selected as possible jump are expanded by decreasing the  percentage that a cell has to have when compared with the maximum of the cells within the radius, where the other consumer probably was. 

is run by >python vector_multiple_consumers_0.4.py name_plot

- agregated_sea_consumption_v5.py Fift instance of the code
acts as a library or the functions developed on the previous version 
"sea-consumption-grid.py" is to call this pakage.
this is a streamlined version to compute the total of sea resources consumed given some input parameters
that where taken as a constant in the previous verions

imput parameters:
land_productivity or max land capacity
number_of_consumers
cnt (class containing all the constants)

output:
sum_of_sea_consumption

is run by being called by sea-consumption-grid.py as
import agregated_sea_consumption_v5 as mc

- sea-consumption-grid.py first instance of the general code to compute under which circumstances there is sea consumption

the code runs trought a set range of parameter spaces and computes the amount of sea resources consumed after X time steps
returns an hisotgram of searesource consumed when varing several combinations of parameters
start with land-production vs number of consumers.

is run by > python sea-consumption-grid.py name_plot



