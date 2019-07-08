Data: data or more accurately data manager that takes charges of managing a dataset, include but not limited to: 1) download data; 2) format data; 3) storage raw data in a database; 4) generate input for training; 5) provide label names for the dataset. It should offer convenient interface to access data for later tasks.

Model: the machine learning model that takes an input and predict an output. It contains essential parameters that define the structure of the prediction model, dedicated functions for the specific model like building the model and preprocessing data. It is a static entity that doesn’t actually do the computing by itself.

