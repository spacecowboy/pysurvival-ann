#include <math.h>
#include <stdio.h>
#include <algorithm>
#include "Statistics.hpp"


/**
 * Note that targets is 2*length, groups is length containing
 * groupCount unique integers from 0 to groupCount -1. groupCounts is
 * groupCount long and sums to length.
 */
double TaroneWareMeanPairwise(const std::vector<double> &targets,
                              std::vector<unsigned int> &groups,
                              std::vector<unsigned int> &groupCounts,
                              const unsigned int length,
                              const TaroneWareType twType) {
  unsigned int i, j, k, pairCount;
  int pair;
  double lastTime, weight;
  bool hasFails = false, hasCens = false;
  const unsigned int groupCount = groupCounts.size();

  // Just guarding
  pairCount = 1;
  if (groupCounts.size() > 1) {
    // Division is safe because this will always be an even number
    pairCount = (groupCounts.size() * (groupCounts.size() - 1)) / 2;
  } else {
    return 0;
  }

  // Initialize count variables
  std::vector<double> fails(groupCount, 0.0);
  std::vector<double> cens(groupCount, 0.0);
  std::vector<double> expectedSum(pairCount, 0.0);
  std::vector<double> observedSum(pairCount, 0.0);
  std::vector<double> varianceSum(pairCount, 0.0);

  std::vector<double> atRisk(groupCount, 0.0);
  std::copy(groupCounts.begin(), groupCounts.begin() + groupCount,
            atRisk.begin());

  double expected, var, totalRisk, totalFail;

  // Times are already ordered (at least we assume so)
  // Initialize lastTime to first time
  lastTime = targets.at(0);
  for (i = 0; i <= length; i++) {
    // If a new time is encountered, remove intermediate censored from risk
    if ((i == length || lastTime != targets.at(2*i)) && hasCens) {
      // If a new time is observed, then only censored at previous
      // times have been seen. We need to update riskgroups.
      for (j = 0; j < groupCount; j++) {
        atRisk.at(j) -= cens.at(j);
        cens.at(j) = 0;
      }
      hasCens = false;
    }
    // When we encounter a new unique time we sum up statistics for previous
    // or we reach the end
    if (i == length || (hasFails && targets.at(2*i) != lastTime)) {
      // All statistics for unique time i-1 done
      // Do group stuff, always comparing j to k since k to j is equivalent
      pair = -1;
      for (j = 0; j < groupCount; j++) {
        // Will skip this for last group, but rest must be done
        for (k = j + 1; k < groupCount; k++) {
          pair++;
          totalRisk = atRisk.at(j) + atRisk.at(k);
          totalFail = fails.at(j) + fails.at(k);
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
            observedSum.at(pair) += weight * fails.at(j);

            // Expected failure count: relative group size * total failures
            expected = (atRisk.at(j) / totalRisk) * (totalFail);

            expectedSum.at(pair) += weight * expected;
            // Variance will also be zero if expected is zero
            if (expected > 0 && totalRisk > 1) {
              // Or we might get a NaN
              var = totalFail * (totalRisk - totalFail) / (totalRisk - 1)
                * atRisk.at(j) / totalRisk
                * (1 - atRisk.at(j) / totalRisk);

              varianceSum.at(pair) += var * pow(weight, 2);
            }
          }
        }
        // Last thing to do is to reset counts again
        // And update risks
        atRisk.at(j) -= (fails.at(j) + cens.at(j));
        fails.at(j) = 0;
        cens.at(j) = 0;
      }
      // hasFails is group independent and so must be updated after all loops
      hasFails = false;
      hasCens = false;
    }

    // Always update statistics, but only before end
    if (i < length){
      // Same .at(failure) time as last observed failure time, just add
      // to statistics But since there might be intermediate censored
      // time, we have to update lastTime also

      if (targets.at(2*i + 1)) {
        // Event
        hasFails = true;
        fails.at(groups.at(i)) += 1;
      } else {
        // Censored
        cens.at(groups.at(i)) += 1;
        hasCens = true;
      }
    }

    // Update lastTime here after all group related updates
    if (i < length) {
      lastTime = targets.at(2*i);
    }
  } // End of loop

  // Test statistic is now simply the mean
  double sum = 0;
  double stat;

  pair = -1;
  for (j = 0; j < groupCount; j++) {
    for (k = j + 1; k < groupCount; k++) {
      pair++;
      stat = pow(observedSum.at(pair) - expectedSum.at(pair), 2);
      stat /= varianceSum.at(pair);

      sum += stat;
    }
  }

  return sum / ((double) pairCount);
}


/**
 * High-low means we only care about the first two groups.
 */
double TaroneWareHighLow(const std::vector<double> &targets,
                         std::vector<unsigned int> &groups,
                         std::vector<unsigned int> &groupCounts,
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
  std::vector<double> fails(groupCountActual, 0.0);
  std::vector<double> cens(groupCountActual, 0.0);

  std::vector<double> atRisk(groupCountActual, 0.0);
  std::copy(groupCounts.begin(), groupCounts.begin() + groupCountActual,
            atRisk.begin());

  double expectedSum=0, observedSum=0, varianceSum=0;

  double expected, var, totalRisk, totalFail;

  // Times are already ordered (at least we assume so)
  // Initialize lastTime to first time
  lastTime = targets.at(0);
  for (i = 0; i <= length; i++) {
    // Ignore other groups
    if (i < length && groups.at(i) > 1) {
      continue;
    }

    // If a new time is encountered, remove intermediate censored from risk
    if ((i == length || lastTime != targets.at(2*i)) && hasCens) {
      // If a new time is observed, then only censored at previous
      // times have been seen. We need to update riskgroups.
      for (g = 0; g < groupCount; g++) {
        atRisk.at(g) -= cens.at(g);
        cens.at(g) = 0;
      }
      hasCens = false;
    }
    // When we encounter a new unique time we sum up statistics for previous
    // or we reach the end
    if (i == length || (hasFails && targets.at(2*i) != lastTime)) {
      // All statistics for unique time i-1 done
      // Do group stuff, always comparing j to k since k to j is equivalent
      totalRisk = atRisk.at(j) + atRisk.at(k);
      totalFail = fails.at(j) + fails.at(k);
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
        observedSum += weight * fails.at(j);

        // Expected failure count: relative group size * total failures
        expected = (atRisk.at(j) / totalRisk) * (totalFail);

        expectedSum += weight * expected;
        // Variance will also be zero if expected is zero
        if (expected > 0 && totalRisk > 1) {
          // Or we might get a NaN
          var = totalFail * (totalRisk - totalFail) / (totalRisk - 1)
            * atRisk.at(j) / totalRisk
            * (1 - atRisk.at(j) / totalRisk);

          varianceSum += var * pow(weight, 2);
        }
      }
      // Last thing to do is to reset counts again
      // And update risks
      for (g = 0; g < groupCount; g++) {
        atRisk.at(g) -= (fails.at(g) + cens.at(g));
        fails.at(g) = 0;
        cens.at(g) = 0;
      }

      // hasFails is group independent and so must be updated after all loops
      hasFails = false;
      hasCens = false;
    }

    // Always update statistics, but only before end
    if (i < length){
      // Same .at(failure) time as last observed failure time, just add
      // to statistics But since there might be intermediate censored
      // time, we have to update lastTime also

      if (targets.at(2*i + 1)) {
        // Event
        hasFails = true;
        fails.at(groups.at(i)) += 1;
      } else {
        // Censored
        cens.at(groups.at(i)) += 1;
        hasCens = true;
      }
    }

    // Update lastTime here after all group related updates
    if (i < length) {
      lastTime = targets.at(2*i);
    }
  } // End of loop


  double stat = pow(observedSum - expectedSum, 2) / varianceSum;
  return stat;
}


double SurvArea(const std::vector<double> &targets,
                std::vector<unsigned int> &groups,
                std::vector<unsigned int> &groupCounts,
                const unsigned int length) {
  unsigned int i;
  double lastTime;

  // Initialize count variables
  double fails, cens, atRisk, surv;

  double prevTime = 0;

  double area = 0;

  // Times are already ordered (at least we assume so)
  // At the start, everyone is at risk
  atRisk = groupCounts.at(0);
  // Nothing has been seen yet
  surv = 1.0;
  fails = cens = 0;
  lastTime = targets.at(0);


  for (i = 0; i <= length; i++) {
    if (surv == 0 || atRisk == 0) {
      // Nothing more can contribute
      break;
    }

    // Ignore other groups
    if (i < length && groups.at(i) != 0) {
      continue;
    }

    // If a new time is encountered, remove intermediate censored from risk
    if ((i < length && lastTime != targets.at(2*i)) && cens > 0) {
      atRisk -= cens;
      cens = 0;
    }

    // When we encounter a new unique time we sum up statistics for previous
    if (i == length || (fails > 0 && targets.at(2*i) != lastTime)) {
      area += surv * (lastTime - prevTime);

      surv *= ((atRisk - fails)/atRisk);
      // Last thing to do is to reset counts again
      atRisk -= (fails + cens);
      fails = cens = 0;
      prevTime = lastTime;
    }

    // Always update statistics, but only before end
    if (i < length){
      if (targets.at(2*i + 1)) {
        // Event
        fails += 1;
      } else {
        // Censored
        cens += 1;
      }

      lastTime = targets.at(2*i);
    }
  } // End loop

  return area;
}


/**
 * 1. Maximize end survival rate^-1: Value between 1 and 10^4
 * 2. Maximize median survival time^-1: 10^4 + 1/m
 */
double RiskGroup(const std::vector<double> &targets,
                 std::vector<unsigned int> &groups,
                 std::vector<unsigned int> &groupCounts,
                 const unsigned int length,
                 const bool findHighRisk) {
  unsigned int i;
  double lastTime;

  // Initialize count variables
  double fails, cens, atRisk, surv;

  double medianTime = -1;
  double lastEventTime = targets.at(0);

  // Times are already ordered (at least we assume so)
  // At the start, everyone is at risk
  atRisk = groupCounts.at(0);
  // Nothing has been seen yet
  surv = 1.0;
  fails = cens = 0;
  lastTime = targets.at(0);

  for (i = 0; i <= length; i++) {
    if (surv == 0 || atRisk == 0) {
      // Nothing more can contribute
      break;
    }

    // Ignore other groups
    if (i < length && groups.at(i) != 0) {
      continue;
    }

    // If a new time is encountered, remove intermediate censored from risk
    if ((i < length && lastTime != targets.at(2*i)) && cens > 0) {
      atRisk -= cens;
      cens = 0;
    }

    // When we encounter a new unique time we sum up statistics for previous
    if (i == length || (fails > 0 && targets.at(2*i) != lastTime)) {
      surv *= ((atRisk - fails)/atRisk);
      // Last thing to do is to reset counts again
      atRisk -= (fails + cens);
      fails = cens = 0;

      // Check median
      if (medianTime < 0 && surv <= 0.5) {
        medianTime = lastTime;
      }
    }

    // Always update statistics, but only before end
    if (i < length){
      if (targets.at(2*i + 1)) {
        // Event
        fails += 1;
        // Last event in group
        lastEventTime = targets.at(2*i);
      } else {
        // Censored
        cens += 1;
      }
      // Last time in group
      lastTime = targets.at(2*i);
    }
  } // End loop

  if (findHighRisk) {
    // See if we've hit zero survival yet
    if (surv > 0.0001) {
      // Want to minimize survival
      return 1.0 / surv;
    } else {
      // Include value for successful survival to get correct sorting
      // Minimize the average between median time and last time
      return 10000.0 + 1.0 / ((medianTime + lastTime) / 2.0);
    }
  } else { // findLowRisk
    // See if we're stuck at zero survival still
    if (surv <= 0.0001) {
      // Then maximize average of median and end time
      return (medianTime + lastTime) / 2.0;
    } else if (surv < 1.0) {
      // Median can never exceed last time, so average above is at most lastTime
      return lastTime + surv;
    } else {
      // Including previous max values
      return lastTime + 1.0 + groupCounts.at(0);
    }
  }
}
