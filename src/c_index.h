
#ifndef C_INDEX_H_
#define C_INDEX_H_

/*
Calculates the C index score of array Y compared to the target array T.
If T == Y, C-index = 1. If Y is totally random, C-index = 0.5.

C index always operates on A[N][2] targets. So total length is actually
length*2. Length is the number of rows. The 2 is implicit.
The Y vector on the other hand is an A[N] array! Be wary of this.
 */
double get_C_index(const double * const Y,
                   const double * const T,
                   const unsigned int length);


/*
  Same as normal C-index, but also returns the values for the individual
patients (e.g. what fraction of comparisons were correct for each patient).
Result is also (1 - C), to conform to ANN training where the error is of
interest, not the performance.

Used in CoxCascadeCorrelation.

C index always operates on A[N][2] targets. So total length is actually
length*2. Length is the number of rows. The 2 is implicit.
The Y vector on the other hand is an A[N] array! Be wary of this.
 */
double getPatError(const double * const Y,
                   const double * const T,
                   const unsigned int length,
                   double * const patError);

#endif /*C_INDEX_H_*/
