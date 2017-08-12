# -*- encoding: utf-8 -*-

import math


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def score(scale=20, speed=1):
	return (lambda w: scale * sigmoid(w * speed))
