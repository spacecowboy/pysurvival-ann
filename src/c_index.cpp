#include "c_index.h"
#include <stdio.h>

#include <vector>

/*
 * C index always operates on A[N][2] targets. So total length is actually
length*2. Length is the number of rows. The 2 is implicit.
The Y vector on the other hand is an A[N] array! Be wary of this.
 */
double get_C_index(const std::vector<double> &Y,
                   const std::vector<double> &T) {
  double total = 0, sum = 0;
#pragma omp parallel default(none) shared(total, sum, Y, T)
  {
    double Tx1, Ty1, Tx0, Ty0, outputsx0, outputsy0;
    unsigned int countx, county;

#pragma omp for schedule(dynamic)
    for (countx = 0; countx < Y.size(); countx++) {
      Tx0 = T.at(countx*2);
      Tx1 = T.at(countx*2 + 1);
      outputsx0 = Y.at(countx);

      for (county = 0; county < Y.size(); county++) {
        if (countx == county)
          continue;

        Ty0 = T.at(county*2);
        Ty1 = T.at(county*2 + 1);
        outputsy0 = Y.at(county);

        if (Tx1 == 1 && Ty1 == 1) {
          // Non-censored, compare with all other non-censored
          // First time seeing thing pair
          if (Tx0 < Ty0) {
#pragma omp atomic
            total++;

            if (outputsx0 < outputsy0) {
#pragma omp atomic
              sum++;
            } else if (outputsx0 == outputsy0) {
#pragma omp atomic
              sum += 0.5;
            }
          }
        } else if (Tx1 == 1) {
          // Non-censored and censored. Compare if
          // Compare noncensored with later censored
          // X noncensored
          if (Tx0 <= Ty0) {
#pragma omp atomic
            total++;

            if (outputsx0 < outputsy0) {
#pragma omp atomic
              sum++;
            } else if (outputsx0 == outputsy0) {
#pragma omp atomic
              sum += 0.5;
            }
          }
        }
      }  // end for
    }  // end for
  }  // end parallel
  if (total == 0) {
    return 0;
  } else {
    return sum / total;
  }
};



/*
  Same as normal C-index, but also returns the values for the individual
patients (e.g. what fraction of comparisons were correct for each patient).
Result is also (1/C), to conform to ANN training where the error is of
interest, not the performance.

Used in CoxCascadeCorrelation.

C index always operates on A[N][2] targets. So total length is actually
length*2. Length is the number of rows. The 2 is implicit.
The Y vector on the other hand is an A[N] array! Be wary of this.
 */
double getPatError(const double * const Y,
                   const double * const T,
                   const unsigned int length,
                   double * const patError)
{
	double total = 0, sum = 0, Tx1, Ty1, Tx0, Ty0, outputsx0, outputsy0;
	unsigned int countx,county;

    double patTotal, patSum;
    double patResult, patAvg = 0;

	for(countx = 0; countx < length; countx++) {
      // x is the patient
      patTotal = 0;
      patSum = 0;
      patResult = 0;

		Tx0 = T[countx*2];
		Tx1 = T[countx*2 + 1];
		outputsx0 = Y[countx];

        //printf("targ: %f out: %f\n", Tx0, outputsx0);

		for(county = 0; county < length; county++) {
			if(countx == county)
			    continue;

			Ty0 = T[county*2];
			Ty1 = T[county*2 + 1];
			outputsy0 = Y[county];

			if(Tx1 == 1 && Ty1 == 1) {
              //Non-censored, compare with all other non-censored
				if (Tx0 < Ty0) {
					total++;
                    patTotal++;
					if (outputsx0 < outputsy0) {
						sum++;
                        patSum++;
					}
                    else if (outputsx0 == outputsy0) {
                        sum += 0.5;
                        patSum += 0.5;
                    }
				}
			}
			else if(Tx1 == 1) { //Non-censored and censored. Compare if
				// Compare noncensored with later censored
				// X noncensored
				if(Tx0 < Ty0) {
					total++;
                    patTotal++;
					if(outputsx0 < outputsy0) {
						sum++;
                        patSum++;
                    }
                    else if (outputsx0 == outputsy0) {
                        sum += 0.5;
                        patSum += 05;
                    }
				}
			}
		}
        // Set value for this patient
        patResult = 1.0;
        if (patSum > 0)
          patResult = 1.0 - patSum / patTotal;

        //        if (patResult > 0 && patTotal == 0 && patSum == 0)
        //  printf("R %f, T %d, S %d\n", patResult, patTotal, patSum);

        patError[countx] = patResult;

        patAvg += patError[countx] / length;
        //printf("patAvg so far: %f\n", patAvg);
    }

    return patAvg;

    /*
    double ci = 0;
    if (sum != 0) {
      ci = sum / total;
    }

    if (ci < 0.0000001)
      return 10000000;
    else
    return 1.0 / ci;*/
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
