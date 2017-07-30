
# ELIH: Explain Machine Learning predictions like I'm a human

A library to translate Machine Learning classifiers predictions in a human understandable and simplified form.

Based on the ELI5 library, ELIH adds to the raw ELI5 outputs several layers of transformations, dictionary, formatting, scoring and interpretors to make raw ML explanations understandable by a non data scientist (your client, a customer facing colleague, etc.).

In some way, it does the opposite of what some feature engineering techniques do: if you dummy-encode your variables, if from series you extract some aggregates (max, mean, ...), ELI5 will explain your predictions based on these "technical" variables. For instance, ELIH allows you to bring business sense back by grouping all related variables, on top of other features (values formatting, variables dictionary, ...).

Still in early stage! But feel free to test & discuss.

## Usage example

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

```python
elih.group(
    explain_prediction(clf, valid_xs[1], vec=vec),
    {
        'Pclass': ['Pclass=1', 'Pclass=2', 'Pclass=3']
    }
)
```

![An explanation with Pclass features regrouped in one place](https://github.com/fvinas/elih/blob/master/doc/example2.png)

## TODO

- improve export to standard Python dict & list
- implement a Matplotlib radar chart
- implement another layer called "interpretors"
- implement a final layer called "agregators" that regroups variables (from any layer + additional ones) and interprations to display
- add a additional rendering layer? automatic sentences?
- provide a config-file like way to write business rules
- support for multiple targets classifiers
- support for regressors


