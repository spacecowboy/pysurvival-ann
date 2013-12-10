#include "ErrorFunctionsSurvival.hpp"
#include <cmath>
#include "global.hpp"
#include <string>

double errorSurvMSE(const double * const Y,
                    const unsigned int length,
                    const unsigned int numOfOutput,
                    const double * const outputs)
{
  unsigned int i, n;
  double error = 0, time, event, output;
  // Evaluate each input set
  // Average over all inputs and number of outputs
  for (i = 0; i < length; i++) {
    for (n = 0; n < numOfOutput; n++) {
      // Plus two because there is an event column as well
      time = Y[2*n];
      event = Y[2*n + 1];
      // No event column in predictions
      output = outputs[n];
      if ((event == 0 && output < time) || event != 0) {
        // Censored event which we are underestimating
        // Or real event
        error += std::pow(output - time, 2.0) / 2.0;
      }
    }
  }
  return error / ((double) length * numOfOutput);
}

void derivativeSurvMSE(const double * const Y,
                       const unsigned int length,
                       const unsigned int numOfOutput,
                       const double * const outputs,
                       const unsigned int index,
                       double * const result)
{
  double time = Y[numOfOutput * index];
  double event = Y[numOfOutput * index + 1];
  double pred = outputs[numOfOutput * index];

  // Only for events or underestimated censored
  if ((event == 0 && pred < time) || event != 0) {
    result[0] = pred - time;
  }
}

// Should come up with better names...
const std::string SURV_A = "SURV_A";
const std::string SURV_B = "SURV_B";
const std::string SURV_C = "SURV_C";
// Indices of events after the given index
const std::string SURV_LATER_EVENTS = "SURV_LATER_EVENTS";
// Probability of dying at each time index
const std::string SURV_PROB = "SURV_PROB";
// Probability for living longer than last event
const std::string SURV_P_AFTER = "SURV_P_AFTER";
// Last event's time
//const std::string SURV_LAST_EVENT_TIME = "SURV_LAST_EVENT_TIME";
// Last member of data: time
const std::string SURV_LAST_TIME= "SURV_LAST_TIME";
// Last member of data: event
const std::string SURV_LAST_EVENT = "SURV_LAST_EVENT";

/**
 * Check if the survival pre-calculations have been performed.
 * Always assumed to be (time, event) in array.
 * Length describes the number of such pairs.
 */
void initSurvCache(const double * const Y,
                   const unsigned int length) {
  // If any is not there, recompute them all
  if (JGN_errorCacheVectorMap.find(SURV_A) == JGN_errorCacheVectorMap.end() ||
      JGN_errorCacheVectorMap.find(SURV_B) == JGN_errorCacheVectorMap.end() ||
      JGN_errorCacheVectorMap.find(SURV_C) == JGN_errorCacheVectorMap.end())
    {
      double atRisk[length];
      double risk[length];
      double surv[length];
      surv[0] = 1.0; // by definition
      double surv_diff = 0;
      int prev_i = 0;
      bool first_event = true;

      // First the risk and survival must be calculated
      for (int i = 0; i < length; i++) {
        double time = Y[2 * i];
        double event = Y[2 * i + 1];

        // init to zero
        atRisk[i] = 0;

        risk[i] = 0;
        // Just so we can sum safely later
        if (i > 0) surv[i] = 0;

        // Fi nd later events
        for (int later = 0; later < length; later++) {
          if (later == i) continue; // Skip itself
          double later_time = Y[2 * later + 1];
          double later_event = Y[2 * later + 1];

          if (later_time > time && later_event == 1) {
            JGN_errorCacheVectorMap[SURV_LATER_EVENTS].push_back(later);
          }
          if (later_time >= time && event == 1) {
            atRisk[i]++;
          }
        }

        // Find last one
        if (JGN_errorCacheVectorMap[SURV_LAST_TIME].empty()) {
          JGN_errorCacheVectorMap[SURV_LAST_TIME].push_back(time);
          JGN_errorCacheVectorMap[SURV_LAST_EVENT].push_back(event);
        }
        else if (time > JGN_errorCacheVectorMap[SURV_LAST_TIME].at(0)) {
          JGN_errorCacheVectorMap[SURV_LAST_TIME][0] = time;
          JGN_errorCacheVectorMap[SURV_LAST_EVENT][0] = event;
        }

        if (event == 1) {
          if (first_event) {
            prev_i = i;
            first_event = false;
          }
          // Calculate risk
          // Risk of division by zero?
          risk[i] = 1.0 / atRisk[i];

          if (i > prev_i) {
            surv[i] = surv[prev_i] + surv_diff;
          }

          surv_diff = -risk[i] * surv[i];
          prev_i = i;

          // Make sure we remember the last event
          /*
          if (JGN_errorCacheVectorMap[SURV_LAST_EVENT_TIME].empty()) {
            JGN_errorCacheVectorMap[SURV_LAST_EVENT_TIME].push_back(time);
          }
          if (time > JGN_errorCacheVectorMap[SURV_LAST_EVENT_TIME].at(0)) {
            JGN_errorCacheVectorMap[SURV_LAST_EVENT_TIME][0] = time;
            }*/
        }
        // Push back non-events (zeros) to maintain length
        JGN_errorCacheVectorMap[SURV_PROB].push_back(risk[i] * surv[i]);
      }
      for (int i = 0; i < length; i++) {
        double time = Y[2 * i];
        double event = Y[2 * i + 1];

        if (event == 1) {
          // Will only be used for censored ones, so push zeros for
          // events
          JGN_errorCacheVectorMap[SURV_P_AFTER].push_back(0);
          JGN_errorCacheVectorMap[SURV_B].push_back(0);
          JGN_errorCacheVectorMap[SURV_A].push_back(0);
          JGN_errorCacheVectorMap[SURV_C].push_back(0);
          continue;
        }

        double sum_prob_later = 0;
        double sum_prob_later_squared_time = 0;
        double sum_prob_later_time = 0;

        for (int later = 0; later < length; later++) {
          if (later == i) continue; // Skip itself

          double later_time = Y[2 * later + 1];
          double later_event = Y[2 * later + 1];

          if (later_time > time && later_event == 1) {
            sum_prob_later += JGN_errorCacheVectorMap[SURV_PROB].at(later);

            sum_prob_later_squared_time +=
              JGN_errorCacheVectorMap[SURV_PROB].at(later) *
              (later_time * later_time);

            sum_prob_later_time +=
              JGN_errorCacheVectorMap[SURV_PROB].at(later) *
              -2 * later_time;
          }
        }

        JGN_errorCacheVectorMap[SURV_P_AFTER].push_back(1.0 - sum_prob_later);
        JGN_errorCacheVectorMap[SURV_B].push_back(sum_prob_later);
        JGN_errorCacheVectorMap[SURV_A].push_back(sum_prob_later_squared_time);
        JGN_errorCacheVectorMap[SURV_C].push_back(sum_prob_later_time);
      }
    }
}

double errorSurvLikelihood(const double * const Y,
                           const unsigned int length,
                           const unsigned int numOfOutput,
                           const double * const outputs)
{
  initSurvCache(Y, length);

  double error = 0;
  //  double last_time = JGN_errorCacheVectorMap[SURV_LAST_EVENT_TIME].at(0);

  for (int i = 0; i < length; i++) {
    double time = Y[numOfOutput * i];
    double event = Y[numOfOutput * i + 1];
    double pred = outputs[numOfOutput * i];
    double local_error = 0;

    if (event == 1) {
     local_error = std::pow(time - pred, 2.0);
    }
    else {
      if (JGN_errorCacheVectorMap[SURV_LATER_EVENTS].size() > 0) {
        // Censored before last event
        // Error for events
        local_error += JGN_errorCacheVectorMap[SURV_A].at(i) +
          pred * (pred * JGN_errorCacheVectorMap[SURV_B].at(i) +
                  JGN_errorCacheVectorMap[SURV_C].at(i));
      }
      // Error due to tail-censored elements
      if (JGN_errorCacheVectorMap[SURV_LAST_EVENT].at(0) == 0 &&
          pred < JGN_errorCacheVectorMap[SURV_LAST_TIME].at(0)) {
        local_error += JGN_errorCacheVectorMap[SURV_P_AFTER].at(i) *
          std::pow(JGN_errorCacheVectorMap[SURV_LAST_TIME].at(0) - pred, 2.0);
      }
    }
    error += local_error;
  }

  return error;
}

void derivativeSurvLikelihood(const double * const Y,
                              const unsigned int length,
                              const unsigned int numOfOutput,
                              const double * const outputs,
                              const unsigned int idx,
                              double * const result)
{
  initSurvCache(Y, length);

  double time = Y[idx];
  double event = Y[idx + 1];
  double pred = outputs[idx];

  // Survival function only cares about first output neuron
  //if (numOfOutput > 1 && idx > 0)
  unsigned int index = idx / numOfOutput;

  // Only first neuron is used
  result[1] = 0;
  if (event == 1) {
    result[0] = 2 * (time - pred);
  }
  else {
    // Later events
    result[0] =
      2 * pred * JGN_errorCacheVectorMap[SURV_B].at(index);
    result[0] += JGN_errorCacheVectorMap[SURV_C].at(index);

    // Tail censored ones
    if (JGN_errorCacheVectorMap[SURV_LAST_EVENT].at(0) == 0 &&
        pred < JGN_errorCacheVectorMap[SURV_LAST_TIME].at(0)) {
          result[0] += JGN_errorCacheVectorMap[SURV_P_AFTER].at(index) *
            2 * (JGN_errorCacheVectorMap[SURV_LAST_TIME].at(0) - pred);
    }
  }
}
