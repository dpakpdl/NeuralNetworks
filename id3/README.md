# python-trees
python implementation of [id3 classification trees](https://en.wikipedia.org/wiki/ID3_algorithm). id3 is a machine learning algorithm for building classification trees developed by Ross Quinlan in/around 1986.

The algorithm is a greedy, recursive algorithm that partitions a data set on the attribute that maximizes information gain. The information gain of attribute A is defined as the difference between the entropy of a data set S and the size weighted average entropy for sub datasets S' of S when split on attribute A. 

## Running the code
Run the code with the python interpreter: 

```python id3.py ./resources/<config.cfg>```

Where config.cfg is a plain text configuration file. The format of the config file is a python abstract syntax tree representing a dict with the following fields:

``
{
   'data_file' : '\\resources\\tennis.csv',
   'data_project_columns' : ['Outlook', 'Temperature', 'Humidity', 'Windy', 'PlayTennis'],
   'target_attribute' : 'PlayTennis'
}
``

You have to specify:
 + relative path to the csv data_file
 + which columns to project from the file (useful if you have a large input file, and are only interested in a subset of columns)
 + the target attribute, that you want to predict.

### Examples
1. tennis.cfg is the 'Play Tennis' example from Machine Learning, by Tim Mitchell, also used by Dr. Lutz Hamel in his lecture notes, both referenced above.
2. credithistory.cfg is the credit risk assement example from [Artificial Intelligence: Structures and Strategies for Complex Problem Solving (6th Edition), Luger](https://www.amazon.com/Artificial-Intelligence-Structures-Strategies-Complex/dp/0321545893), see Table 10.1 & Figure 10.14 (full text is available online asof 11/19/2017).  

### Results

![results](https://github.com/dpakpdl/NeuralNetworks/blob/master/id3/resources/results.png)

## TODO
- represent result in tree structure
- Add code to classify data.
- Add code to prune rules (C4.5 modifications)
