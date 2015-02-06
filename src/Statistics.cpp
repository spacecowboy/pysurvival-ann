#include <math.h>
#include <stdio.h>
#include <algorithm>
#include "Statistics.hpp"


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
                              const TaroneWareType twType) {
  unsigned int i, j, k, pairCount;
  double lastTime, weight;
  bool hasFails = false, hasCens = false;

  // Just guarding
  pairCount = 1;
  if (groupCount > 1) {
    // Division is safe because this will always be an even number
    pairCount = (groupCount * (groupCount - 1)) / 2;
  }

  // Initialize count variables
  double fails[groupCount], cens[groupCount], atRisk[groupCount],
    expectedSum[pairCount], observedSum[pairCount],
    varianceSum[pairCount];

  std::fill(fails, fails + groupCount, 0);
  std::fill(cens, cens + groupCount, 0);
  std::fill(observedSum, observedSum + pairCount, 0);
  std::fill(expectedSum, expectedSum + pairCount, 0);
  std::fill(varianceSum, varianceSum + pairCount, 0);

  for (i = 0; i < groupCount; i++) {
    atRisk[i] = groupCounts[i];
  }

  double expected, var, totalRisk, totalFail;

  // Times are already ordered (at least we assume so)
  // Initialize lastTime to first time
  lastTime = targets[0];
  for (i = 0; i <= length; i++) {
    // If a new time is encountered, remove intermediate censored from risk
    if ((i == length || lastTime != targets[2*i]) && hasCens) {
      // If a new time is observed, then only censored at previous
      // times have been seen. We need to update riskgroups.
      for (j = 0; j < groupCount; j++) {
        atRisk[j] -= cens[j];
        cens[j] = 0;
      }
      hasCens = false;
    }
    // When we encounter a new unique time we sum up statistics for previous
    // or we reach the end
    if ((hasFails && targets[2*i] != lastTime) || i == length) {
      // All statistics for unique time i-1 done
      // Do group stuff, always comparing j to k since k to j is equivalent
      for (j = 0; j < groupCount; j++) {
        // Will skip this for last group, but rest must be done
        for (k = j + 1; k < groupCount; k++) {
          totalRisk = atRisk[j] + atRisk[k];
          totalFail = fails[j] + fails[k];
          // If total risk = 0, then none of the sums will have more terms added
          // If we reach the end and have only censored, then this means stop
          if (totalRisk > 0 && totalFail > 0) {
            // Weight depends on choice of statistic.
            switch (twType) {
            case TaroneWareType::GEHAN:
              weight = totalRisk;
              break;
            case TaroneWareType::TARONEWARE:
              weight = sqrt(totalRisk);
              break;
            case TaroneWareType::LOGRANK:
            default:
              weight = 1.0;
              break;
            }

            // Sum up all failures observed
            observedSum[j * groupCount + k] += weight * fails[j];

            // Expected failure count: relative group size * total failures
            expected = (atRisk[j] / totalRisk) * (totalFail);

            expectedSum[j * groupCount + k] += weight * expected;
            // Variance will also be zero if expected is zero
            if (expected > 0 && totalRisk > 1) {
              // Or we might get a NaN
              var = totalFail * (totalRisk - totalFail) / (totalRisk - 1)
                * atRisk[j] / totalRisk
                * (1 - atRisk[j] / totalRisk);

              varianceSum[j * groupCount + k] += var * pow(weight, 2);
            }
          }
        }
        // Last thing to do is to reset counts again
        // And update risks
        atRisk[j] -= (fails[j] + cens[j]);
        fails[j] = 0;
        cens[j] = 0;
      }
      // hasFails is group independent and so must be updated after all loops
      hasFails = false;
      hasCens = false;
    }

    // Always update statistics, but only before end
    if (i < length){
      // Same [failure] time as last observed failure time, just add
      // to statistics But since there might be intermediate censored
      // time, we have to update lastTime also

      if (targets[2*i + 1]) {
        // Event
        hasFails = true;
        fails[groups[i]] += 1;
      } else {
        // Censored
        cens[groups[i]] += 1;
        hasCens = true;
      }
    }

    // Update lastTime here after all group related updates
    if (i < length) {
      lastTime = targets[2*i];
    }
  } // End of loop

  // Test statistic is now simply the mean
  double sum = 0;
  double stat;
  for (j = 0; j < groupCount; j++) {
    for (k = j + 1; k < groupCount; k++) {
      stat = pow(observedSum[j * groupCount + k] - expectedSum[j * groupCount + k], 2)
        / varianceSum[j * groupCount + k];

      sum += stat;
    }
  }

  return sum / ((double) pairCount);
}


/**
 * High-low means we only care about the first two groups.
 */
double TaroneWareHighLow(const double * const targets,
                         const unsigned int * const groups,
                         const unsigned int * const groupCounts,
                         const unsigned int length,
                         const unsigned int groupCountActual,
                         const TaroneWareType twType) {
  const unsigned int j=0, k=1;
  unsigned int i, g, groupCount;
  double lastTime, weight;
  bool hasFails = false, hasCens = false;

  // Only look at first two groups
  if (groupCountActual >= 2) {
    groupCount = 2;
  } else {
    // Can't work with one group
    return 0;
  }

  // Initialize count variables
  double fails[groupCount], cens[groupCount], atRisk[groupCount],
    expectedSum=0, observedSum=0, varianceSum=0;

  std::fill(fails, fails + groupCount, 0);
  std::fill(cens, cens + groupCount, 0);

  for (i = 0; i < groupCount; i++) {
    atRisk[i] = groupCounts[i];
  }

  double expected, var, totalRisk, totalFail;

  // Times are already ordered (at least we assume so)
  // Initialize lastTime to first time
  lastTime = targets[0];
  for (i = 0; i <= length; i++) {
    // If a new time is encountered, remove intermediate censored from risk
    if ((i == length || lastTime != targets[2*i]) && hasCens) {
      // If a new time is observed, then only censored at previous
      // times have been seen. We need to update riskgroups.
      for (g = 0; g < groupCount; g++) {
        atRisk[g] -= cens[g];
        cens[g] = 0;
      }
      hasCens = false;
    }
    // When we encounter a new unique time we sum up statistics for previous
    // or we reach the end
    if ((hasFails && targets[2*i] != lastTime) || i == length) {
      // All statistics for unique time i-1 done
      // Do group stuff, always comparing j to k since k to j is equivalent
      totalRisk = atRisk[j] + atRisk[k];
      totalFail = fails[j] + fails[k];
      // If total risk = 0, then none of the sums will have more terms added
      // If we reach the end and have only censored, then this means stop
      if (totalRisk > 0 && totalFail > 0) {
        // Weight depends on choice of statistic.
        switch (twType) {
        case TaroneWareType::GEHAN:
          weight = totalRisk;
          break;
        case TaroneWareType::TARONEWARE:
          weight = sqrt(totalRisk);
          break;
        case TaroneWareType::LOGRANK:
        default:
          weight = 1.0;
          break;
        }

        // Sum up all failures observed
        observedSum += weight * fails[j];

        // Expected failure count: relative group size * total failures
        expected = (atRisk[j] / totalRisk) * (totalFail);

        expectedSum += weight * expected;
        // Variance will also be zero if expected is zero
        if (expected > 0 && totalRisk > 1) {
          // Or we might get a NaN
          var = totalFail * (totalRisk - totalFail) / (totalRisk - 1)
            * atRisk[j] / totalRisk
            * (1 - atRisk[j] / totalRisk);

          varianceSum += var * pow(weight, 2);
        }
      }
      // Last thing to do is to reset counts again
      // And update risks
      for (g = 0; g < groupCount; g++) {
        atRisk[g] -= (fails[g] + cens[g]);
        fails[g] = 0;
        cens[g] = 0;
      }

      // hasFails is group independent and so must be updated after all loops
      hasFails = false;
      hasCens = false;
    }

    // Always update statistics, but only before end
    if (i < length){
      // Same [failure] time as last observed failure time, just add
      // to statistics But since there might be intermediate censored
      // time, we have to update lastTime also

      if (targets[2*i + 1]) {
        // Event
        hasFails = true;
        fails[groups[i]] += 1;
      } else {
        // Censored
        cens[groups[i]] += 1;
        hasCens = true;
      }
    }

    // Update lastTime here after all group related updates
    if (i < length) {
      lastTime = targets[2*i];
    }
  } // End of loop


  double stat = pow(observedSum - expectedSum, 2) / varianceSum;
  return stat;
}
