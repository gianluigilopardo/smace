# SMACE --- Semi-Model-Agnostic Contextual Explainer
Python code for [*SMACE: A New Method for the Interpretability of Composite Decision Systems*](https://arxiv.org/abs/2111.08749).

SMACE is a new interpretability method for hybrid decision systems that aggregate multiple machine learning models through decision rules.
It combines a geometric approach (for decision rules) with existing interpretability solutions (for machine learning models) to generate explanations based on feature importance.

The [evaluation](https://github.com/gianluigilopardo/smace/tree/main/evaluation) folder contains the experiments in the manuscript.

The file [example.ipynb](https://github.com/gianluigilopardo/smace/blob/main/example.ipynb "example.ipynb") is an executable notebook in which the use of SMACE is shown.

## Usage
First, one must define the decision-making system, *i.e.*, a ```DM``` object.
To define it, you need a set of rules in JSON format, a list of models, and a [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

### 1. Define your set of rules 
The rules must be defined in a JSON object, resulting in Python lists/dictionaries.
Each ```rule``` is a dictionary with two fields: ```conditions``` and ```decision```.
The latter is the output of the decision process, if the rule is satisfied.
A condition is defined by the triple ```(name, operator, value)```:
* ```name``` is the variable referred to;
* ```operator``` can be ```geq``` ($\geq$), ```gt``` ($>$), ```leq``` ($\leq$), ```lt``` ($<$);
* ```value``` is the cutoff.

As an example, let us say our set of variables includes four features: $x_1$, $x_2$, $x_3$, $x_4$, and two models: ```model_1``` and ```model_2```.
The JSON with two rules ```rule1``` and ```rule2``` can be as follow:
```
{"rule1": {"conditions": [{"name": "x2",
                             "operator": "geq",
                             "value": 0.6},
                            {"name": "x3",
                             "operator": "geq",
                             "value": 0.25},
                            {"name": "model_1",
                             "operator": "geq",
                             "value": 1},
                            {"name": "model_2",
                             "operator": "leq",
                             "value": 50}],
            "decision": "decision1"},
"rule2": {"conditions": [{"name": "x4",
                             "operator": "geq",
                             "value": 0.1},
                            {"name": "model_1",
                             "operator": "geq",
                             "value": 0.2},
                            {"name": "x1",
                             "operator": "geq",
                             "value": 0.1},
                            {"name": "x4",
                             "operator": "leq",
                             "value": 0.9}],
            "decision": "decision2"}
}
```
Once defined, to read a JSON file one can use the ```json``` ([docs here](
https://docs.python.org/3/library/json.html)) to read it:
```
import json
with open('rules.json', 'r') as fp:
    rules_json = json.load(fp)
```

### 2. Define your list of models 
A model can be any function that works on a subset of the original data, with a numerical output. ```DM``` needs a ```Model``` object initialized as ```Model(predictive_function, model_name, data)```, where
* ```predictive_function``` is the function that produces the output. In the case of a ```sklearn``` model ```m``` for regression (resp., for classification), for instance, it corresponds to ```m.predict``` (resp., ```m.predict_proba```);
* ```model_name``` is the name used in the rules to refer to the output of the model;
* ```data``` is the ```pandas.DataFrame``` to which the model is applied.

For example, assuming we have a dataset ```X``` and two targets ```y1``` and ```y2```, we can proceed as follows:
```
from smace.models import Model

lm = linear_model.LinearRegression()
lm.fit(X,y1)

xgb = xgboost.XGBClassifier()
xgb.fit(X,y2)

model_1 = Model(lm.predict, 'model_1', df)
model_2 = Model(xgb.predict_proba, 'model_2', df)

models_list = [model_1, model_2]
```

### 3. Define the ```DM``` object
Having the rules ```rules_json```, the list of models ```models_list``` and the input dataset ```df```, you can construct the ```DM``` object as
```
from smace.decisions import DM
dm = DM(rules_json, models_list, df)
```
To get the decision explicitly for an example, we use the ```make_decision()``` function:
```
example = np.random.rand(4)
decision = dm.make_decision(example, verbose=True)
```
```
Output:
	Rule(s) ['rule1'] triggered.
	Decision(s) ['decision1'] made.
```

### Apply SMACE
Once the configuration is complete, you can use SMACE to explain the decisions of the defined system.

Let us say we want to explain why for the example above ```rule2``` was not triggered:
```
from smace.explainer import Smace
explainer = Smace(dm)

explanation = explainer.explain(example, 'rule2')
```
```explanation``` contains all the information computed by SMACE. The following methods can be applied:
* ```explanation.table()``` and ```explanation.bar()```  to obtain the overall contributions of the input features as tables or bars, respectively;
* ```explanation.rule_table()``` and ```explanation.rule_bar()```  to get the contributions of all variables in the rule as tables or bars, respectively;
* ```explanation.model_table('mod')``` and ```explanation.model_bar('mod')``` to get the importance of input features to the model named ```'mod'```.

It is possible to specify the maximum number of variables to display through the ```num_features``` parameters.
