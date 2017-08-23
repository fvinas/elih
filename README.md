
ELIH: Explain Machine Learning predictions like I'm a human
===========================================================

A library to translate Machine Learning classifiers predictions in a human understandable and simplified form.

Based on [ELI5](https://github.com/TeamHG-Memex/eli5), ELIH adds to the raw ELI5 explanation several layers of transformations, translation, formatting, scoring and interpretation to make ML predictions understandable by a non data scientist (e.g. your client, a customer facing colleague, etc.).

ELIH works at prediction level (and not at model level, like the `feature importance`-like figures).

In a way, it does the opposite of what some feature engineering techniques do: if you dummy encode your variables or if from series you compute some aggregates (max, mean, ...) that are used as input variables by the model, ELI5 will explain your predictions based on these "internal" or "technical" variables. What ELIH does it bringing human or business sense back to the explanation, by grouping all related variables, formatting their value, giving them a human-readable label...

Along with ELI5, ELIH may also be used to debug and improve machine learning models, since it may help understand which features worked, imagine new features to add, better understand false predictions, ...

ELIH has been tested to be compatible with Python 2.7+ and Python 3.5+ (via `six` and `future`).

Still in early stage! But feel free to test & discuss.

Example
-------

### ELI5 output

```python

# Should be tested in the context of the XGBoost + Titanic ELI5 example
# https://github.com/TeamHG-Memex/eli5/blob/master/docs/source/_notebooks/xgboost-titanic.rst
# chapter 4: "Explaning predictions"

from eli5 import explain_prediction
import elih

explain_prediction(clf, valid_xs[1], vec=vec)
```

![An explanation with raw features](https://github.com/fvinas/elih/blob/master/doc/example1.png)

This explanation is the output of ELI5. You have everything, but it's mostly a technical view for you datascientist. You cannot really use it with external stakeholders (your clients, customer facing people, etc.).

### ELIH output

ELIH allows you to bring additional human logic layers on top of this raw output:

![An output of ELIH](https://github.com/fvinas/elih/blob/master/doc/example2.png)

```python
from eli5 import explain_prediction
from eli5 import formatters

idx = 45

exp = elih.HumanExplanation(
    explanation=explain_prediction(clf, valid_xs[idx], vec=vec),  # an ELI5 Explanation object
    rules_layers=[{
        # 1st layer -- variables back from dummy encoding
        'Pclass': 'Pclass=*',
        'Sex': 'Sex=*',
        'Source port': 'Embarked=*',
        'Cabin': 'Cabin=*',
        'Ticket #': 'Ticket=*'
    }, {
        # 2nd layer -- grouping advanced features
        'Family': ['Parch', 'SibSp'],
        'Cabin & location': ['Cabin', 'Fare', 'Ticket #', 'Pclass'],
        'Person': ['Sex', 'Age']
    }],
    additional_features=valid_xs_no_transform[idx],  # raw input data (before preprocessing)
    dictionary={
        'Sex': {
            'label': 'Sex',
            'value_from': 'Sex',
            'formatter': elih.formatters.mapper({
                'female': 'female',
                'male': 'male'
            })
        },
        'Fare': {
            'label': u'Ticket fare',
            'formatter': elih.formatters.value_simplified(decimals=0, unit="$")
        },
        'Cabin': {
            'label': u'Cabin number',
            'value_from': 'Cabin',
            'formatter': elih.formatters.text()
        },
        'Age': {
            'label': u'Age',
            'formatter': (lambda a: '{} yrs'.format(int(a)))
        },
        'Parch': {
            'label': u'# parents & children onboard',
            'formatter': elih.formatters.integer()
        },
        'SibSp': {
            'label': u'# siblings & spouse onboard',
            'formatter': elih.formatters.integer()
        },
        'Source port': {
            'label': u"Embarkment port",
            'value_from': 'Embarked',
            'formatter': elih.formatters.mapper({
                'C': 'Cherbourg',
                'Q': 'Queenstown',
                'S': 'Southampton'
            })
        },
        'Ticket #': {
            'label': u'Ticket number',
            'formatter': elih.formatters.text(),
            'value_from': 'Ticket'
        },
        'Pclass': {
            'label': u'Passenger class',
            'value_from': 'Pclass',
            'formatter': elih.formatters.text()
        }
    },
    scoring=(lambda x: 10-10*elih.scoring.sigmoid(3*x)),
    interpretors={
        'PASSENGER_ALONE': {
            'assert': lambda v: v['Parch'] == 0 and v['SibSp'] == 0,
            'interpretation': lambda v: 'Passenger is travelling alone',
            'not_interpretation': lambda v: 'Passenger is not travelling alone'
        },
        'PASSENGER_ABOVE_50': {
            'assert': lambda v: v['Age'] > 50,
            'interpretation': lambda v: 'Passenger is above 50 ({})'.format(v['Age']['formatted_value'])
        }
    }
)
```

Features
--------

### Grouping layers

ELIH allows you to define one or several layers of "grouping", where different model input variables are regrouped together to form a new explanatory variable, whose contribution is the sum of all underlying variables contributions.

A common way to use it to defined two layers (as shown in the example above):
- a first, technical layer to get human variables back from technical encodings required by the model (dummy encoding, aggregates computation, ...)
- a second, business layer to group together variables that belong to the same *domain* or that are (closely) related to each other

Grouping rules are given to ELIH using the `rules_layers` argument.

### Enrichment with additional variables

ELIH allows you to bring additional variables into the explanation (ie not used by the model).

It's often used jointly with the grouping feature: you may use it to bring a value back from dummy encoded variables, or display historical values of events when the model only uses aggregates (for instance the number of events in a given time range).

Additional variables are provided to ELIH using `additional_features`.

### Dictionary

ELIH allows you to provide a dictionary for all the variables (ie not only the raw ones, but also the additional, external ones and the groups). This dicionary provides the following features:
- adding labels to the variables
- formatting their values using custom or standard formatters (see below)

Variables dictionary is provided to ELIH using `dictionary`.

### Scoring

ELIH allows you to add a scoring layer that translates ELI5 contributions into your custom scale.
A standard, sigmoid-based score is provided ready to use.

The score function is passed to ELIH using `scoring`.

### Interpretation rules

ELIH allows you to define custom interpretation rules in the form of lambda functions; when matched, these rules generate interpretation text. They are specified using argument `interpretors`.

### Export

ELIH displays fancy HTML tables to easily understand your predictions in a Jupyter Notebook, but allows you to export its output in standard Python objects (dict, list, ...) thru its `to_dict` method so that you can directly use its output in a production workflow or in another application.

Usage
-----

`elih.HumanExplanation(explanation, rules_layers, additional_features=None, dictionary=None, scoring=None, interpretors=None)`: returns a `elih.HumanExplanation` object.


`elih.HumanExplanation` constructor arguments are:

- `explanation` - an ELI5 `Explanation` object (typically the output of `explain_prediction`)
- `rules_layers` - `list` of `dict` to define rules layers (or a `dict` if only one layer).
    
    Each layer `dict` describes the variables grouping that will be performed by the layer.
    Keys are the newly created variables while values are the ones used to create it.
    The first layer can only use input variables from ELI5 explanation, while subsequent layers can use variables created by the previous layers.
    When a variable is not mentioned in the rules, it's kept as it was.
    You have two ways to specify underlying variables to create a group:

    - exact match: a list of variables names (e.g `['Cabin', 'Fare', 'Ticket #', 'Pclass']`)
    - filewise pattern matching: (e.g `Sex=*`)

    By default a grouped variable has no value, but ELIH provides a way to set one (using the `dictionary` argument).

    It's best to avoid re-using variables names that are already used elsewhere because it will lead to unexpected behaviours (variable created last would overwrite previous content).

- `additional_features` - (optional, defaults to `None`) a `dict` to provide ELIH with additional variables and their values, on top of the ones used as input values by the model. These variables can then be used to
fill grouped variables from the rules layers with a value, can be used by the interpretation rules and can be manipulated by the `dictionary` (formatting, label, ...).

- `dictionary` - (optional, defaults to `None`) a `dict` whose key is the name of a variable to provide ELIH with additional information on. This variable can either be a model input variable given through the `explanation`, an additional variable provided with `additional_features`, or a new grouped variable defined through the `rules_layers` (from any layer).

    Dictionary for variable can take several arguments - all optional:

    - `label` - a `string` to provide ELIH with a human understandable label for the variable
    - `value_from` - only applicable for grouped variables created in the rules layers. A `string` to provide ELIH with the name of an additional variable (from `additional_features`) from which the grouped variable will take its value (especially useful when considering *dummy decoding* - see example above)
    - `formatter` - provides ELIH with a formatter for this variable. A formatter is a lambda function whose role is to pretty print the value of a variable. ELIH comes with several standard formatters for common cases (displaying units, mapping of values, simplifying values with `k`, `M`, `B`, ...) but any custom formatter can also be used (see example above).

    ELIH formatters include:

     ```python
     elih.formatters.text
     elih.formatters.integer
     elih.formatters.value(decimals=1, unit="", sign="")
     elih.formatters.percent(decimals=1)
     elih.formatters.delta_percent(decimals=1)
     elih.formatters.value_simplified(decimals=1, unit="", prefixes=['k', 'M', 'B'], sign="")
     elih.formatters.mapper(dictionary)
     ```

- `scoring` - ELIH provides a simple scoring system that allows you to easily generate a custom score from the ELI5 contribution weights. The `scoring` argument expects a lambda function as the scoring function. This function will transform the contribution weights into a score.

    You may implement it by yourself using the *sigmoid* function from `elih.scoring.sigmoid`, or use a basic score implementation like `elih.scoring.score`:

    ```python
    def sigmoid(x):
	    return 1 / (1 + math.exp(-x))

    def score(scale=20, speed=1):
	    return (lambda w: scale * sigmoid(w * speed))
    ```

- `interpretors` - ELIH allows you to implement custom *interpretors* so that it can automatically match (or not) interpretation defined by rules. You have to provide ELIH with a `dict` of interpretation rules.

    Each rule is defined by a key, and by three characteristics given through a `dict`:

    ```python
	interpretors={
        'PASSENGER_ALONE': {
            'assert': lambda v: v['Parch'] == 0 and v['SibSp'] == 0,
            'interpretation': lambda v: 'Passenger is travelling alone',
            'not_interpretation': lambda v: 'Passenger is not travelling alone ({} other family members)'.format(v['Parch'] + v['SibSp'])
        },
        ...
    }
    ```

    - `assert` - this lambda function will be evaluated to match a rule. Its only argument will be a `dict` of all variables ELIH knows about (being input variables, additional variables or variables created by rules layers). An assertion lambda function is expected to return `True` or `False`.

    - `interpretation` - this lambda function is expected to return a string, being the human interpretation in case the interpretation rule defined by `assert` matched. This lambda function only argument is the same as `assert`, a `dict` with all variables known from ELIH (see example above).

    - `not_interpretation` - (optional) this is the opposite of `interpretation`. Will be called in case the interpretation rule did not match. If missing, there will be no interpretation in this case.


Once you have a `HumanExplanation` object, you can either display it (via `__repr__` or `_repr_html_`) or export it to use its output in another piece of code, using its `to_dict` method.


Roadmap
-------

- implement a radar chart
- implement a end layer of "aggregators" that groups variables (from any layer + additional ones) and interprations to display
- add a additional rendering layer? automatic sentences?
- support for multiple targets classifiers
- support for regressors
- unit tests
- move formatters to an external PyFormatters library

Authors
=======

Originally created and maintained by [Fabien Vinas](https://github.com/fvinas)

License
=======

Apache 2 Licensed. See LICENSE for full details.
