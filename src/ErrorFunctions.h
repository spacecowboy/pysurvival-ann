#ifndef _ERRORFUNCTIONS_H_
#define _ERRORFUNCTIONS_H_

double SSEDeriv(double target, double output);
double *SSEDerivs(double *target, double *output, int length);
double SSE(double target, double output);
double *SSEs(double *target, double *output, int length);


#endif /* _ERRORFUNCTIONS_H_ */
