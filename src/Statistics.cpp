#include <math.h>
#include <stdio.h>
#include <algorithm>
#include "Statistics.hpp"


/**
 * Note that targets is 2*length, groups is length containing
 * groupCount unique integers from 0 to groupCount -1. groupCounts is
 * groupCount long and sums to length.
 */
double logRankStatistic(const double * const targets,
                        const unsigned int * const groups,
                        const unsigned int * const groupCounts,
                        const unsigned int length,
                        const unsigned int groupCount) {
  unsigned int i, j, k;
  double lastTime;
  bool hasFails = false, hasCens = false;

  // Initialize count variables
  double fails[groupCount], cens[groupCount], atRisk[groupCount],
    expectedSum[groupCount*groupCount], observedSum[groupCount],
    varianceSum[groupCount*groupCount];

  std::fill(fails, fails + groupCount, 0);
  std::fill(cens, cens + groupCount, 0);
  std::fill(observedSum, observedSum + groupCount, 0);
  std::fill(expectedSum, expectedSum + groupCount*groupCount, 0);
  std::fill(varianceSum, varianceSum + groupCount*groupCount, 0);

  for (i = 0; i < groupCount; i++) {
    //fails[i] = 0;
    //cens[i] = 0;
    //printf("\nGroup %i has %i members\n", i, groupCounts[i]);
    atRisk[i] = groupCounts[i];
    //observedSum[i] = 0;
    //for (j = i + 1; j < groupCount; j++) {
    //  expectedSum[i * groupCount + j] = 0;
    //  varianceSum[i * groupCount + j] = 0;
    //}
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
        //printf("\nattempt with %f", cens[j]);
        atRisk[j] -= cens[j];
        cens[j] = 0;
      }
      hasCens = false;
    }
    // When we encounter a new unique time we sum up statistics for previous
    // or we reach the end
    if ((hasFails && targets[2*i] != lastTime) || i == length) {
      //printf("\nUnique Time: %f", lastTime);
      // All statistics for unique time i-1 done
      // Do group stuff, always comparing j to k since k to j is equivalent
      for (j = 0; j < groupCount; j++) {
        // Sum up all failures observed
        observedSum[j] += fails[j];

        // Will skip this for last group, but rest must be done
        for (k = j + 1; k < groupCount; k++) {
          totalRisk = atRisk[j] + atRisk[k];
          totalFail = fails[j] + fails[k];
          // If total risk = 0, then none of the sums will have more terms added
          // If we reach the end and have only censored, then this means stop
          if (totalRisk > 0 && totalFail > 0) {
            //printf("\nTotalRisk: %f + %f = %f", atRisk[j], atRisk[k], totalRisk);
            //printf("\nTotalFail: %f + %f = %f", fails[j], fails[k], totalFail);

            // Expected failure count: relative group size * total failures
            expected = (atRisk[j] / totalRisk) * (totalFail);
            //printf("\nExpect: %f", expected);
            expectedSum[j * groupCount + k] += expected;
            // Variance, includes expected as a term
            // Not sure how to handle this correctly... TODO
            if (expected > 0 && totalRisk > 1) {
              // Or we might get a NaN
              var = (atRisk[k] / (totalRisk - 1)) * (1 - totalFail/totalRisk) * expected;
              //printf("Var: (%f / %f) * %f * %f = %f\n", atRisk[k], (totalRisk - 1),
              //       (1 - totalFail/totalRisk), expected, var);
              //printf("\nVariance: %f", var);
              varianceSum[j * groupCount + k] += var;
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
        //printf("\nUpdating cens");
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
  // n will be groupcount * (groupcount - 1) / 2
  double n = 0;
  //printf("\nChecking pairwise...\n");
  for (j = 0; j < groupCount; j++) {
    for (k = j + 1; k < groupCount; k++) {
      n++;
      //printf("\nObservedSum = %f", observedSum[j]);
      //printf("\nExpectedSum = %f", expectedSum[j * groupCount + k]);
      //printf("\nVarianceSum = %f", varianceSum[j * groupCount + k]);

      //printf("Pre stat: %f - %f = %f **2 = %f\n", observedSum[j],
      //       expectedSum[j * groupCount + k],
      //       observedSum[j] - expectedSum[j * groupCount + k],
      //       pow(observedSum[j] - expectedSum[j * groupCount + k], 2));

      stat = pow(observedSum[j] - expectedSum[j * groupCount + k], 2)
        / varianceSum[j * groupCount + k];
      //printf("Pairwise stat %i-%i: %f (%i vs %i) [%f]\n", j, k, stat,
      //       groupCounts[j], groupCounts[k], varianceSum[j * groupCount + k]);
      sum += stat;
    }
  }

  /*
  // This is just for testing
  for (j = 0; j < groupCount; j++) {
    // Return 0 if any groups are less than some value
    if (groupCounts[j] < 30) {
      printf("\nReturning zero fitness...\n");
      return 0;
    }
    }*/

  //printf("\nReturning fitness %f / %f = %f\n", sum, n, sum / n);
  return sum / n;
}
