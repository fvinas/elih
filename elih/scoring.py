# -*- encoding: utf-8 -*-

import math

# TODO: implement a default scoring system based on a sigmoid
# propose a simple way to map it to a 10-notches scale


def sigmoid(x):
	'''Classical sigmoid function
	'''
	return 1 / (1 + math.exp(-x))


def score(scale=20, speed=1):
	return (lambda w: scale * sigmoid(w * speed))
