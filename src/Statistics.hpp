#ifndef _STATISTICS_H_
#define _STATISTICS_H_

/**
 * This changes the choice of weight in the TaroneWareStatistic
 */
enum class TaroneWareType { LOGRANK, // weight = 1
    GEHAN, // weight = totalRisk
    TARONEWARE}; // weight = sqrt(totalRisk)


/**
 * Note that targets is 2*length, groups is length containing
 * groupCount unique integers from 0 to groupCount -1. groupCounts is
 * groupCount long and sums to length.
 */
double TaroneWareMeanPairwise(const double * const targets,
                              const unsigned int * const groups,
                              const unsigned int * const groupCounts,
                              const unsigned int length,
                              const unsigned int groupCount,
                              const TaroneWareType twType);

/**
 * Note that targets is 2*length, groups is length containing
 * groupCount unique integers from 0 to groupCount -1. groupCounts is
 * groupCount long and sums to length.
 */
double TaroneWareHighLow(const double * const targets,
                         const unsigned int * const groups,
                         const unsigned int * const groupCounts,
                         const unsigned int length,
                         const unsigned int groupCount,
                         const TaroneWareType twType);


/**
 * Return the area under the KaplanMeier graph. This is an
 * (under-)estimate of the mean survival time.
 */
double SurvArea(const double * const targets,
                const unsigned int * const groups,
                const unsigned int * const groupCounts,
                const unsigned int length);

#endif /* _STATISTICS_H_ */
