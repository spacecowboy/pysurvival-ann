#ifndef _STATISTICS_H_
#define _STATISTICS_H_

#include <vector>

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
double TaroneWareMeanPairwise(const std::vector<double> &targets,
                              std::vector<unsigned int> &groups,
                              std::vector<unsigned int> &groupCounts,
                              const unsigned int length,
                              const TaroneWareType twType);

/**
 * Note that targets is 2*length, groups is length containing
 * groupCount unique integers from 0 to groupCount -1. groupCounts is
 * groupCount long and sums to length.
 */
double TaroneWareHighLow(const std::vector<double> &targets,
                         std::vector<unsigned int> &groups,
                         std::vector<unsigned int> &groupCounts,
                         const unsigned int length,
                         const unsigned int groupCount,
                         const TaroneWareType twType);


/**
 * Return the area under the KaplanMeier graph. This is an
 * (under-)estimate of the mean survival time.
 */
double SurvArea(const std::vector<double> &targets,
                std::vector<unsigned int> &groups,
                std::vector<unsigned int> &groupCounts,
                const unsigned int length,
                const bool invert);

/**
 * Goal here is to maximize the risk groups. for HighRisk groups,
 * it first tries to bring survival rate to zero (10^-4). If successful,
 * it then tries to minimize the median survival time.
 * For LowRisk groups, it first tries to bring survival rate to 1.0.
 * If successful, it then tries to maximize the group size.
 */
double RiskGroup(const std::vector<double> &targets,
                 std::vector<unsigned int> &groups,
                 std::vector<unsigned int> &groupCounts,
                 const unsigned int length,
                 const bool findHighRisk);

#endif /* _STATISTICS_H_ */
