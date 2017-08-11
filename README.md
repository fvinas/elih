
ELIH: Explain Machine Learning predictions like I'm a human
===========================================================

A library to translate Machine Learning classifiers predictions in a human understandable and simplified form.

Based on the ELI5 library, ELIH adds to the raw ELI5 outputs several layers of transformations, dictionary, formatting, scoring and interpretors to make raw ML explanations understandable by a non data scientist (your client, a customer facing colleague, etc.).

In a way, it does the opposite of what some feature engineering techniques do: if you dummy-encode your variables or from series you compute some aggregates (max, mean, ...) that are used as input variables by the model, ELI5 will explain your predictions based on these "internal" or "technical" variables. ELIH allows you to bring business sense back to the explanation, by grouping all related variables, formatting their value, giving them a human-readable label...

Along with ELI5, you may use it also to debug and improve your ML workflow: understand which features worked, imagine new features to add, better understand false predictions, ...

Still in early stage! But feel free to test & discuss.

Example
-------

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
            #'formatter': elih.formatters.text(),
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

### Enrichment with additional variables

ELIH allows you to bring additional variables into the explanation (ie not used by the model).

It's often used jointly with the grouping feature: you may use it to bring a value back from dummy encoded variables, or display historical values of events when the model only uses aggregates (for instance the number of events in a given time range).

### Dictionary

ELIH allows you to provide a dictionary for all the variables (ie not only the raw ones, but also the additional, external ones and the groups). This dicionary provides the following features:
- adding labels to the variables
- formatting their values using custom or standard formatters (see below)

### Scoring

ELIH allows you to add a scoring layer that translates ELI5 contributions into your custom scale.
A standard, sigmoid-based score is provided ready to use.

### Interpretation rules

ELIH allows you to define custom interpretation rules in the form of lambda functions; when matched, these rules generate interpretation text.

### Export

ELIH displays fancy HTML tables to easily understand your predictions in a Jupyter Notebook, but allows you to export its output in standard Python objects (dict, list, ...) so that you can directly use its output in a production workflow or in another application.

Usage
-----

*TODO*

Roadmap
-------

- implement a Matplotlib radar chart
- implement a final layer called "agregators" that regroups variables (from any layer + additional ones) and interprations to display
- add a additional rendering layer? automatic sentences?
- provide a config-file like way to write business rules
- support for multiple targets classifiers
- support for regressors

Authors
=======

Originally created and maintained by [Fabien Vinas](https://github.com/fvinas)

License
=======

Apache 2 Licensed. See LICENSE for full details.
