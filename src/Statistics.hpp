#ifndef _STATISTICS_H_
#define _STATISTICS_H_

/**
 * Note that targets is 2*length, groups is length containing
 * groupCount unique integers from 0 to groupCount -1. groupCounts is
 * groupCount long and sums to length.
 */
double TaroneWareStatistic(const double * const targets,
                           const unsigned int * const groups,
                           const unsigned int * const groupCounts,
                           const unsigned int length,
                           const unsigned int groupCount);

#endif /* _STATISTICS_H_ */
