/*
 * activationfunctions.h
 *
 *  Created on: 7 sep 2012
 *      Author: jonas
 */

#ifndef ACTIVATIONFUNCTIONS_H_
#define ACTIVATIONFUNCTIONS_H_

/*
 * Linear, y = x, derivative = 1
 */
double linear(double x);
double linearDeriv(double y);

/*
 * Sigmoid, y = 1 / (1 + exp(-x)), deriv = y * (1 - y)
 */
double sigmoid(double x);

double sigmoidDeriv(double y);

/*
 * Hyperbole, y = tanh(x), deriv = (1 - y) * (1 + y)
 */
double hyperbole(double x);

double hyperboleDeriv(double y);

#endif /* ACTIVATIONFUNCTIONS_H_ */
