Folder for code related to the costal resources project

- resources_costal_plain.py fisrt isntance of the code, basic generation and consumption of resources

is run by >python resources_costal_plain_0.1.py 


- matrix_resources_costal_plain_0.2.py second instance of the code, extends the previous consumption and production of resources to a NxM matrix and plots the result in a dinamic movie showing the matrix as cells and plots a graph of all the sea resources consumed. 

is run by >python matrix_resources_costal_plain.py name_plot


- vector_resources_costal_plain.py Third instance of the code,  modifies the previous code to only deal with a vector of consumption, simplifying the sea/land divided consumption of the previous "matrix" version. It also uses a different rutine of choosing where the  consumer will jump, previously was random, now it selects the cells within a radius r around itself that have up to a 80% of the cell containing the maximum of resources, and jumps randomly to any of these cells. 

is run by >python vector_resources_costal_plain_0.3.py name_plot

- vector_multiple_consumers.py Forth instance of the code, it adds several consumers to the vector. The jumping strategy is designed to avoid that two consumers fall in the same cell. If they do, the cells selected as possible jump are expanded by decreasing the  percentage that a cell has to have when compared with the maximum of the cells within the radius, where the other consumer probably was. 

is run by >python vector_multiple_consumers_0.4.py name_plot

