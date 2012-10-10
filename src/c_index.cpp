#include "c_index.h"

/*
 * C index always operates on A[N][2] targets. So total length is actually
length*2. Length is the number of rows. The 2 is implicit.
The Y vector on the other hand is an A[N] array! Be wary of this.
 */
double get_C_index(double *Y, double *T, unsigned int length)
{
	double total = 0, sum = 0, Tx1, Ty1, Tx0, Ty0, outputsx0, outputsy0;
	unsigned int countx,county;

	for(countx = 0; countx < length; countx++) {
		Tx0 = T[countx*2];
		Tx1 = T[countx*2 + 1];
		outputsx0 = Y[countx];

		for(county = 0; county < length; county++) {
			if(countx == county)
			    continue;

			Ty0 = T[county*2];
			Ty1 = T[county*2 + 1];
			outputsy0 = Y[county];

			if(Tx1 == 1 && Ty1 == 1) { //Non-censored, compare with all other non-censored
				if (Tx0 < Ty0) {
					total++;
					if (outputsx0 < outputsy0) {
						sum++;
					}
				}
			}
			else if(Tx1 == 1) { //Non-censored and censored. Compare if
				// Compare noncensored with later censored
				// X noncensored
				if(Tx0 < Ty0) {
					total++;
					if(outputsx0 < outputsy0)
						sum++;
				}
			}
		}
	}

	sum /= total;
    return sum;
};

/*
static PyMethodDef methods[] = {
	{"derivative_beta", // Function name, as seen from python
	derivative_beta, // actual C-function name
	METH_VARARGS, // positional (no keyword) arguments
	NULL}, // doc string for function
	{"get_slope", get_slope, METH_VARARGS, NULL},
	{"get_C_index", get_C_index, METH_VARARGS, NULL},
	{"get_weighted_C_index", get_weighted_C_index, METH_VARARGS, NULL},
};


// This bit is needed to be able to import from python

PyMODINIT_FUNC initcox_error_in_c()
{
	Py_InitModule("cox_error_in_c", methods);
}
*/
