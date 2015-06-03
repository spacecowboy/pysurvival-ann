#include "MatrixNetwork.hpp"
#include "activationfunctions.hpp"
#include <stdio.h>
#include <limits> // max int
#include "Random.hpp"
#include "GeneticNetwork.hpp"
#include "GeneticMutation.hpp"
#include "RPropNetwork.hpp"
#include "ErrorFunctionsSurvival.hpp"
#include "ErrorFunctions.hpp"
#include "Statistics.hpp"

#include <iostream>
#include <cmath>
#include <assert.h>
#include <omp.h>


void matrixtest() {
  std::cout << "\nMatrixTest...";

    //printf("\nConstructing 2, 2, 2");
    MatrixNetwork m(2, 0, 2);

    //printf("\nBIAS_INDEX = %d", m.BIAS_INDEX);
    //printf("\nINPUT_RANGE = %d - %d", m.INPUT_START, m.INPUT_END);
    //printf("\nHIDDEN_RANGE = %d - %d", m.HIDDEN_START, m.HIDDEN_END);
    //printf("\nOUTPUT_RANGE = %d - %d", m.OUTPUT_START, m.OUTPUT_END);

    //printf("\n\nINPUT_COUNT = %d", m.INPUT_COUNT);
    //printf("\nHIDDEN_COUNT = %d", m.HIDDEN_COUNT);
    //printf("\nOUTPUT_COUNT = %d", m.OUTPUT_COUNT);

    //printf("\n\nLENGTH = %d", m.LENGTH);

    for (unsigned int i = 0; i < m.LENGTH; i++) {
      //printf("\n\nNeuron %d, default func = %d", i, m.actFuncs.at(i));
      m.actFuncs.at(i) =
        (ActivationFuncEnum) (i % 3);

      for (unsigned int j = 0; j < m.LENGTH; j++) {
        // printf("\nConnection %d = %d (%f) (default)",
        //       j,
        //       m.conns.at(m.LENGTH * i + j),
        //       m.weights.at(m.LENGTH * i + j));
        m.conns.at(m.LENGTH * i + j) = 1;
        m.weights.at(m.LENGTH * i + j) = j + 1;
        //printf("\nConnection %d = %d (%f) (now)",
        //      j,
        //      m.conns.at(m.LENGTH * i + j),
        //      m.weights.at(m.LENGTH * i + j));
      }
    }

    m.setHiddenActivationFunction(TANH);
    m.setOutputActivationFunction(LINEAR);

    // printf("\n\nTANH = %d, LOGSIG = %d\n", TANH, LOGSIG);
    for (unsigned int i = m.HIDDEN_START; i < m.HIDDEN_END; i++) {
      // printf("\nHIDDEN(%d).actFunc = %d", i, m.actFuncs.at(i));
    }
    // printf("\n\nTANH = %d, LOGSIG = %d\n", TANH, LOGSIG);
    for (unsigned int i = m.OUTPUT_START; i < m.OUTPUT_END; i++) {
      // printf("\nOUTPUT(%d).actFunc = %d", i, m.actFuncs.at(i));
    }

    std::vector<double> outputs(m.OUTPUT_COUNT, 0.0);
    std::vector<double> inputs(m.INPUT_COUNT, 0.0);

    for (unsigned int i = m.INPUT_START; i < m.INPUT_END; i++) {
      inputs.at(i) = 2;
    }

    m.output(inputs.begin(), outputs.begin());

    for (unsigned int i = 0; i < m.OUTPUT_COUNT; i++) {
      printf("\nNetwork output.at(%d): %f", i, outputs.at(i));
    }

    std::cout << "\nMatrixTest Done.";
}

void randomtest() {
  std::cout << "\nRandomTest...";
  Random r(0);

  printf("\n  Uniform: %f", r.uniform());

  Random rr;

  printf("\n  Normal: %f", rr.normal());
  printf("\n  Geometric(10): %d", rr.geometric(10));

  std::cout << "\n  UINT lim: " << std::numeric_limits<unsigned int>::max();

  std::cout << "\n  UINT: " << r.uint();

  std::cout << "\n  Uniform number: " << rr.uniformNumber(1, 10);
  std::cout << "\nRandomTest Done.";
}

void geneticTest1() {
  printf("\nGeneticTest1...");

  vector<GeneticNetwork*> population;

  unsigned int extras, populationSize = 50;
#pragma omp parallel
  {
#pragma omp master
    {
      extras = 4 * omp_get_num_threads();
    }
    // End master
  }
  // End parallel

  // Pre-allocate space
  population.reserve(populationSize + extras);

  Random rand;
  GeneticMutator mutator(rand);

  GeneticNetwork base(20, 3, 2);
  mutator.randomizeNetwork(base, 1.0);


  for (unsigned int i = 0; i < populationSize + extras; i++) {
    GeneticNetwork *pNet = new GeneticNetwork(base.INPUT_COUNT,
                                              base.HIDDEN_COUNT,
                                              base.OUTPUT_COUNT);
    // Base it on the all-mother
    pNet->cloneNetwork(base);

    population.insert(population.begin() + i, pNet);
  }

  for (unsigned int gen = 0; gen < 100; gen++) {
#pragma omp parallel default(none) shared(base, population)
    {
      GeneticNetwork *pChild;
      Random rand;
      GeneticMutator mutator(rand);
#pragma omp for
      for (unsigned int threadChild = 0; threadChild < 50; threadChild++) {
#pragma omp critical
        {
          // Pop a network
          pChild = population.at(0);
          population.erase(population.begin());
        } // End critical

        mutator.mutateWeights(*pChild, 1.0, 1.0);
        mutator.mutateConns(*pChild, 1.0);

#pragma omp critical
        {
          population.insert(population.begin(), pChild);
        } // End critical
      } // End for
    }// End parallel
  }


  //mutator.randomizeNetwork(net1, 1.0);

  //mutator.randomizeNetwork(net2, 1.0);

  // printf("\nWeight diff: %f vs %f
//\nConn diff: %d vs %d
//\nActF diff: %d vs %d",
//         net1.weights[9],
//         net2.weights[9],
//         net1.conns[9],
//         net2.conns[9],
//         net1.actFuncs[3],
//         net2.actFuncs[3]);

  // printf("\nCloning...");
  //net2.cloneNetwork(net1);

  // printf("\nWeight diff: %f vs %f
//\nConn diff: %d vs %d
//\nActF diff: %d vs %d",
//         net1.weights[9],
  //       net2.weights[9],
    //     net1.conns[9],
    //     net2.conns[9],
     //    net1.actFuncs[3],
      //   net2.actFuncs[3]);

  vector<GeneticNetwork*>::iterator netIt;
  for (netIt = population.begin(); netIt < population.end();
       netIt++) {
    delete *netIt;
  }



  printf("\nGeneticTest1 Done.");
}

void geneticXOR() {
  std::cout << "\nGeneticXOR...";

  GeneticNetwork net(2, 5, 1);
  net.setGenerations(100);
  net.setWeightMutationChance(0.5);
  net.setWeightMutationFactor(0.3);
  net.connsMutationChance = 0.5;
  net.actFuncMutationChance = 0.5;
  net.setCrossoverChance(0.6);

  // define inputs
  std::vector<double> X = {0,0,
                           0,1,
                           1,0,
                           1,1};
  // printf("\nbah %f", X[7]);

  // define targets
  std::vector<double> Y = {0, 1, 1, 0};
  // printf("\nbah %f", Y[2]);

  net.learn(X, Y, 4);

  std::vector<double> preds(1, 0.0);

   std::cout << "\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X.begin() + 2 * i, preds.begin());
     std::cout << X[2*i] << " "<< X[2*i + 1]
              << " : " << preds[0] << "\n";
  }

  // Print structure
   std::cout << "\n\nWeights";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
     std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      std::cout << " " << net.weights.at(j + i*net.LENGTH);
    }
  }

   std::cout << "\n\nConss";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
     std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      std::cout << " " << net.conns.at(j + i*net.LENGTH);
    }
  }

   std::cout << "\n\nActFuncs";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    std::cout << "\nN" << i << ": " << net.actFuncs.at(i);
  }

  std::cout << "\nGeneticXOR Done.";
}

void geneticXORDroput() {
  std::cout << "\nGeneticXORDropout...";

  GeneticNetwork net(2, 5, 1);
  net.setGenerations(100);
  net.setWeightMutationChance(0.5);
  net.setWeightMutationFactor(0.3);
  net.connsMutationChance = 0.5;
  net.actFuncMutationChance = 0.5;
  net.setCrossoverChance(0.6);
  net.hiddenDropoutProb = 0.9;

  // define inputs
  std::vector<double> X = {0,0,
                           0,1,
                           1,0,
                           1,1};
  // printf("\nbah %f", X[7]);

  // define targets
  std::vector<double> Y = {0, 1, 1, 0};
  // printf("\nbah %f", Y[2]);

  net.learn(X, Y, 4);

  std::vector<double> preds(1, 0.0);

   std::cout << "\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X.begin() + 2 * i, preds.begin());
     std::cout << X[2*i] << " "<< X[2*i + 1]
              << " : " << preds[0] << "\n";
  }

  // Print structure
   std::cout << "\n\nWeights";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
     std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      std::cout << " " << net.weights.at(j + i*net.LENGTH);
    }
  }

   std::cout << "\n\nConss";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
     std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      std::cout << " " << net.conns.at(j + i*net.LENGTH);
    }
  }

   std::cout << "\n\nActFuncs";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    std::cout << "\nN" << i << ": " << net.actFuncs.at(i);
  }

  std::cout << "\nGeneticXORDropout Done.";
}


void rpropsurvlik() {
  std::cout << "\nRPropSurvLikTest...";

  Random r;
  RPropNetwork net(2, 8, 2);

  // Set up a feedforward structure
  for (unsigned int i = net.OUTPUT_START; i < net.LENGTH; i++) {
    for (unsigned int j = net.HIDDEN_START; j < net.HIDDEN_END; j++) {
      net.conns.at(net.LENGTH * i + j) = 1;
      net.weights.at(net.LENGTH * i + j) = r.normal();
    }
    // Also activate self
    net.conns.at(net.LENGTH * i + i) = 1;
  }
  for (unsigned int i = net.HIDDEN_START; i < net.HIDDEN_END; i++) {
    for (unsigned int j = 0; j < net.BIAS_END; j++) {
      net.conns.at(net.LENGTH * i + j) = 1;
      net.weights.at(net.LENGTH * i + j) = r.normal();
    }
    // Also activate self
    net.conns.at(net.LENGTH * i + i) = 1;
  }
  net.setHiddenActivationFunction(TANH);
  net.setOutputActivationFunction(LINEAR);

  std::vector<double> X = {0,0,
                           0,1,
                           1,0,
                           1,1,
                           0,0,
                           0,1,
                           1,0,
                           1,1};

  // define targets
  // initial censored point which segfaulted at some time
  std::vector<double> Y = {0, 0,
                           1, 1,
                           1, 1,
                           0, 1,
                           0, 1,
                           0.5, 0,
                           0.5, 0,
                           0, 0};

  // Print structure
  // std::cout << "\n\nWeights before";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    // std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      // std::cout << " " << net.weights[j + i*net.LENGTH];
    }
  }

  net.setMaxEpochs(100);
  net.setMaxError(0.001);
  net.setErrorFunction(ErrorFunction::ERROR_SURV_LIKELIHOOD);
  if (0 < net.learn(X, Y, 8)) {
    throw "Shit hit the fan";
  }

  // Print structure
  // std::cout << "\n\nWeights after";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    // std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      // std::cout << " " << net.weights[j + i*net.LENGTH];
    }
  }

  std::vector<double> preds(2, 0);
  // std::cout << "\n\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X.begin() + 2 * i, preds.begin());
    // std::cout << X[2*i] << " "<< X[2*i + 1]
       //       << " : " << std::round(preds[0])
        //      << " (" << Y[i] << ")"<< "\n";
  }

  std::cout << "\nRPropSurvLikTest Done.";
}

void rproptest() {
  std::cout << "\nRPropTest...";

  Random r;
  RPropNetwork net(2, 5, 1);

  // Set up a feedforward structure
  for (unsigned int i = net.OUTPUT_START; i < net.LENGTH; i++) {
    for (unsigned int j = net.HIDDEN_START; j < net.HIDDEN_END; j++) {
      net.conns.at(net.LENGTH * i + j) = 1;
      net.weights.at(net.LENGTH * i + j) = r.normal();
    }
    // Also activate self
    net.conns.at(net.LENGTH * i + i) = 1;
  }
  for (unsigned int i = net.HIDDEN_START; i < net.HIDDEN_END; i++) {
    for (unsigned int j = 0; j < net.BIAS_END; j++) {
      net.conns.at(net.LENGTH * i + j) = 1;
      net.weights.at(net.LENGTH * i + j) = r.normal();
    }
    // Also activate self
    net.conns.at(net.LENGTH * i + i) = 1;
  }
  net.setHiddenActivationFunction(TANH);
  net.setOutputActivationFunction(LOGSIG);

  // xor
    // define inputs
  std::vector<double> X = {0,0,
                           0,1,
                           1,0,
                           1,1};
  // printf("\nbah %f", X[7]);

  // define targets
  std::vector<double> Y = {0, 1, 1, 0};

  // Print structure
  std::cout << "\n\nConns before";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j <= i; j++) {
      std::cout << " " << net.conns.at(j + i*net.LENGTH);
    }
  }

  net.setMaxEpochs(1000);
  //net.setMaxError(0.01);
  //net.setMinErrorFrac(0.001);
  if ( 0 < net.learn(X, Y, 4)) {
    throw "Shit hit the fan";
  }

  // Print structure
  // std::cout << "\n\nWeights after";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    // std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      // std::cout << " " << net.weights.at(j + i*net.LENGTH);
    }
  }

  bool success = true;
  std::vector<double> preds(1, 0);
  // std::cout << "\n\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X.begin() + 2 * i, preds.begin());

    std::cout << "\n  " << X[2*i] << " "<< X[2*i + 1]
              << " : " << preds[0]
              << " (" << Y[i] << ")";

    double diff = preds[0] - Y[i];
    if (diff < 0) diff = -diff;

    if (diff > 0.1) success = false;
  }
  assert(success);

  std::cout << "\nRPropTest Done.";
}

void rpropdropouttest() {
  std::cout << "\nRPropDropoutTest...";

  Random r;
  RPropNetwork net(2, 5, 1);
  net.hiddenDropoutProb = 0.9;

  // Set up a feedforward structure
  for (unsigned int i = net.OUTPUT_START; i < net.LENGTH; i++) {
    for (unsigned int j = net.HIDDEN_START; j < net.HIDDEN_END; j++) {
      net.conns.at(net.LENGTH * i + j) = 1;
      net.weights.at(net.LENGTH * i + j) = r.normal();
    }
    // Also activate self
    net.conns.at(net.LENGTH * i + i) = 1;
  }
  for (unsigned int i = net.HIDDEN_START; i < net.HIDDEN_END; i++) {
    for (unsigned int j = 0; j < net.BIAS_END; j++) {
      net.conns.at(net.LENGTH * i + j) = 1;
      net.weights.at(net.LENGTH * i + j) = r.normal();
    }
    //Start with zero here to test dropout
    net.conns.at(net.LENGTH * i + i) = 0;
  }
  net.setHiddenActivationFunction(TANH);
  net.setOutputActivationFunction(LOGSIG);

  // xor
    // define inputs
  std::vector<double> X = {0,0,
                           0,1,
                           1,0,
                           1,1};
  // printf("\nbah %f", X[7]);

  // define targets
  std::vector<double> Y = {0, 1, 1, 0};

  // Print structure
  std::cout << "\n\nConns before";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    std::cout << "\nN" << i << "(" << net.conns.at(i + i*net.LENGTH) << ")" << ":";
    for (unsigned int j = 0; j <= i; j++) {
      std::cout << " " << net.conns.at(j + i*net.LENGTH);
    }
  }

  net.setMaxEpochs(30000);
  net.setMaxError(0.0);
  net.setMinErrorFrac(0.01);
  if ( 0 < net.learn(X, Y, 4)) {
    throw "Shit hit the fan";
  }

  // Print structure
  std::cout << "\n\nConns after";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
     std::cout << "\nN" << i << "(" << net.conns.at(i + i*net.LENGTH) << ")" << ":";
     assert(net.conns.at(i + i*net.LENGTH) == 1);
    for (unsigned int j = 0; j < i; j++) {
      std::cout << " " << net.conns.at(j + i*net.LENGTH);
    }
  }

  std::vector<double> preds(1, 0);
  // std::cout << "\n\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X.begin() + 2 * i, preds.begin());

    std::cout << "\n  " << X[2*i] << " "<< X[2*i + 1]
              << " : " << preds[0]
              << " (" << Y[i] << ")";

    double diff = preds[0] - Y[i];
    if (diff < 0) diff = -diff;

  }

  std::cout << "\nRPropDropoutTest Done.";
}

void rpropalloctest() {
  std::cout << "\nRPropAllocTest...";

  Random r;
  RPropNetwork net(2, 8, 2);

  // Set up a feedforward structure
  for (unsigned int i = net.OUTPUT_START; i < net.LENGTH; i++) {
    for (unsigned int j = net.HIDDEN_START; j < net.HIDDEN_END; j++) {
      net.conns.at(net.LENGTH * i + j) = 1;
      net.weights.at(net.LENGTH * i + j) = r.normal();
    }
  }
  for (unsigned int i = net.HIDDEN_START; i < net.HIDDEN_END; i++) {
    for (unsigned int j = 0; j < net.BIAS_END; j++) {
      net.conns.at(net.LENGTH * i + j) = 1;
      net.weights.at(net.LENGTH * i + j) = r.normal();
    }
  }
  net.setHiddenActivationFunction(TANH);
  net.setOutputActivationFunction(LOGSIG);

  // xor
  // define inputs
  unsigned int limit = 100000;

  std::vector<double> X(2*limit, 0);
  // define targets
  std::vector<double> Y(2*limit, 0);

  // Not interested in training result

  net.setMaxEpochs(1);
  net.setMaxError(0.001);

  net.setErrorFunction(ErrorFunction::ERROR_MSE);
  if ( 0 < net.learn(X, Y, limit)) {
    throw "Shit hit the fan";
  }

  std::cout << "\nRPropAllocTest Done.";
}

// Takes a few minutes to run
void survcachealloctest() {
  std::cout << "\nSurvCacheAllocTest...";

  unsigned int limit = 100000;
  std::vector<double> Y(2*limit, 0);

  SurvErrorCache *cache = new SurvErrorCache();

  cache->verifyInit(Y, limit);

  delete cache;

  std::cout << "\nSurvCacheAllocTest Done";
}


// Assert that the difference between a and b is (absolute value) less
// than 10^-10
void assertSame(const double a, const double b)
{
  double diff = a - b;
  if (diff < 0) diff = -diff;

  assert(diff < 0.00000000001);
}

void testSurvCache() {
  std::cout << "\nTestSurvCache...";
  // Need some known data to work with.
  const unsigned int length = 35, sortedIndex = 17;
  unsigned int index;
  // These are disordered
  std::vector<double> targets = {
    0.20648482, 1.,
    0.20824676, 1.,
    0.21543471, 0.,
    0.21772184, 0.,
    0.22132258, 1.,
    0.22375735, 0.,
    0.22591362, 0.,
    0.23053736, 0.,
    0.23824816, 0.,
    0.10281301, 0.,
    0.10810285, 0.,
    0.1100851, 1.,
    0.11439226, 0.,
    0.13502698, 1.,
    0.13922102, 0.,
    0.14320695, 1.,
    0.14496066, 0.,
    0.14599487, 1.,
    0.14793153, 1.,
    0.19055548, 0.,
    0.19444517, 1.,
    0.19481425, 0.,
    0.19634443, 1.,
    0.198517, 0.,
    0.14902403, 0.,
    0.15076501, 0.,
    0.15624607, 0.,
    0.16141124, 0.,
    0.16218441, 0.,
    0.17173613, 1.,
    0.17444673, 1.,
    0.17716717, 0.,
    0.18045819, 0.,
    0.18104029, 0.,
    0.18960431, 1.
};

  // First we need to get the sorted indices
  std::vector<unsigned int> sortedIndices;
  getIndicesSortedByTime(targets, length, sortedIndices);
  assert(sortedIndices.size() == length);
  index = sortedIndices.at(sortedIndex);
  // Make sure index is pointing to correct time
  assertSame(targets[2*index], 0.17716717);
  assert(targets[2*index+1] == 0);

  // First method should be get prob, this is used in the rest
  std::vector<double> probs;
  std::vector<double> survival;
  getProbsAndSurvival(targets, length, sortedIndices, probs, survival);

  // This should be a vector with length equal to entire times vector squared.
  assert(probs.size() == length);
  assert(survival.size() == length);
  // Probs at censored events is zero
  double cumprob = 0;
  for (unsigned int i = 0; i < length; i++) {
    if (targets[2*i + 1]) {
      assert(probs.at(i) > 0);
    }
    else {
      assert(probs.at(i) == 0);
    }
    // Calculate for first patient, which has unscaled probs
    cumprob += probs.at(i);
  }
  // Cumulative sum must > 0 and <= 1
  assert(cumprob > 0);
  assert(cumprob <= 1.0);
  std::cout << "\n  cumprob: " << cumprob;
  assertSame(cumprob, 0.632038916092);

  // Prob after method is just 1.0 - sum(Probs)
  double scaledProb;
  std::vector<unsigned int>::const_iterator it;
  it = sortedIndices.begin() + sortedIndex;
  scaledProb = getScaledProbAfter(targets, probs, survival,
                                  sortedIndices,
                                  it);
  assert(scaledProb >= 0);
  assert(scaledProb <= 1);
  std::cout << "\n  scaled_i: " << scaledProb;
  assertSame(scaledProb, 0.487334887335);

  double Ai;
  Ai = getPartA(targets, probs, survival,
                sortedIndices, it);
  assert(Ai > 0);
  std::cout << "\n  Ai: " << Ai;
  assertSame(Ai, 0.0215826999321);

  double Bi;
  Bi = getPartB(targets, probs, survival,
                sortedIndices, it);
  assert(Bi > 0);
  std::cout << "\n  Bi: " << Bi;
  assertSame(Bi, 0.512665112665);

  double Ci;
  Ci = getPartC(targets, probs, survival,
                sortedIndices, it);
  assert(Ci < 0);
  std::cout << "\n  Ci: " << Ci;
  assertSame(Ci, -0.210069338856);

  double pred, e, d;
  double time = targets[2 * index];
  double lastTime = targets[2 * (*(sortedIndices.end() - 1))];

  pred = 0.1 * time;
  e = getLikelihoodError(time, pred, lastTime,
                         Ai, Bi, Ci, scaledProb);
  assert (e >= 0);
  std::cout << "\n  Eless: " << e;
  assertSame(e, 0.0417229793877);

  d = getLikelihoodDeriv(time, pred, lastTime,
                         Bi, Ci, scaledProb);
  assert (d < 0);
  std::cout << "\n  Dless: " << d;
  assertSame(d, -0.406849185279);

  pred = 3 * time;
  e = getLikelihoodError(time, pred, lastTime,
                         Ai, Bi, Ci, scaledProb);
  assert (e >= 0);
  std::cout << "\n  Emore: " << e;
  assertSame(e, 0.0547552731939);

  d = getLikelihoodDeriv(time, pred, lastTime,
                         Bi, Ci, scaledProb);
  assert (d > 0);
  std::cout << "\n  Dmore: " << d;
  assertSame (d, 0.334895224155);



  std::cout << "\nTestSurvCache Done.";
}

void errorTests()
{
  std::cout << "\nTestErrors...";

  double x = 0.0;
  double y = 2.0;

  for (unsigned int rows = 1; rows < 10; rows++)
  {
    for (unsigned int cols = 1; cols < 10; cols++)
    {
      // MSE is 0.5 * (x - y)^2

      std::vector<double> outputs(rows*cols, 0);
      std::vector<double> targets(rows*cols, 0);
      std::vector<double> errors(rows*cols, 0);

      // init
      for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
          outputs[i * cols + j] = x;
          targets[i * cols + j] = y;
        }
      }

      getAllErrors(ErrorFunction::ERROR_MSE,
                   targets, rows, cols,
                   outputs, errors);

      for (unsigned int j = 0; j < cols; j++) {
        assertSame(errors[j], (0.5*(x-y)*(x-y)));
      }
    }
  }

  std::cout << "\nTestErrors done.";
}

void derivTests()
{
  std::cout << "\nTestDerivs...";

  double x = 0.0;
  double y = 2.0;

  for (unsigned int rows = 1; rows < 10; rows++)
  {
    for (unsigned int cols = 1; cols < 4; cols++)
    {
      // MSE is 0.5 * (x - y)^2
      std::vector<double> outputs(rows*cols, 0);
      std::vector<double> targets(rows*cols, 0);
      std::vector<double> derivs(rows*cols, 0);

      // init
      for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
          outputs.at(i * cols + j) = x;
          targets.at(i * cols + j) = y;
        }
      }

      for (unsigned int i = 0; i < rows; i++) {
        unsigned int index = i * cols;
        getDerivative(ErrorFunction::ERROR_MSE,
                      targets, rows, cols,
                      outputs, index, derivs.begin() + index);
        for (unsigned int j = 0; j < cols; j++) {
          //std::cout << i << "," << j << " d: " << derivs.at(index + j) << "\n";;
          assertSame(derivs.at(index + j), (x-y));
        }
      }
    }
  }

  std::cout << "\nTestDerivs done.";
}

void survMSETests()
{
  std::cout << "\nTestSurvMSE...";

  unsigned int rows = 10;
  unsigned int cols = 2;

  unsigned int i;

  std::vector<double> outputs(rows*cols, 0);
  std::vector<double> targets(rows*cols, 0);
  std::vector<double> errors(rows*cols, 0);

  for (i = 0; i < rows; i++)
  {
      // Targets are always zero
      targets[i * cols] = 0;
      // Check censored
      targets[i * cols + 1] = 0;
      // Even indices underestimate
      if (i % 2 == 0)
      {
        outputs[i * cols] = -1;
      }
      else // Odd overestimate
      {
        outputs[i * cols] = 1;
      }
  }

  getAllErrors(ErrorFunction::ERROR_SURV_MSE,
               targets, rows, cols,
               outputs, errors);

  for (i = 0; i < rows; i++) {
      // Even indices underestimate
      if (i % 2 == 0)
      {
        assertSame(errors[i * cols], 0.5 * (1.0 - 0.0));
      }
      else // overestimate
      {
        assertSame(errors[i * cols], 0.0);
      }
  }

  std::cout << "\nTestSurvMSE done.";
}

void survLikTests()
{
  std::cout << "\nTestSurvLik...";

  unsigned int rows = 20;
  unsigned int cols = 2;

  unsigned int i;

  std::vector<double> outputs(rows*cols, 0);
  std::vector<double> targets(rows*cols, 0);
  std::vector<double> errors(rows*cols, 0);

  //////////////
  // First I want to test uncensored points
  // Then we go on to censored first
  // censored middle
  // censored last
  //////////////

  for (i = 0; i < rows; i++)
  {
      // Targets
      targets.at(i * cols) = i;
      targets.at(i * cols + 1) = 1.0;
      // Even indices underestimate
      if (i % 2 == 0)
      {
        outputs.at(i * cols) = i - 2.0;
      }
      else // Odd overestimate
      {
        outputs.at(i * cols) = i + 2.0;
      }
  }

  // Censor first point
  targets.at(0 + 1) = 0;

  // Censor two in the middle
  targets.at(10 * cols + 1) = 0;
  targets.at(11 * cols + 1) = 0;

  // Censor last two
  targets.at((rows - 2) * cols + 1) = 0;
  targets.at((rows - 1) * cols + 1) = 0;

  getAllErrors(ErrorFunction::ERROR_SURV_LIKELIHOOD,
               targets, rows, cols,
               outputs, errors);

  for (i = 0; i < rows; i++) {
    printf("\n  E: %f", errors.at(i * cols));
      //std::cout << "\ne: " << errors.at(i * cols) << "\n";
    //assertSame(errors.at(i * cols), 4.0);
  }

  std::cout << "\nTestSurvLik done.";
}


void testSoftmax() {
  std::cout << "\nTestSoftmax...";

  for (int out_count = 1; out_count < 10; out_count++) {

    MatrixNetwork m(2, 0, out_count);

    //printf("\nBIAS_INDEX = %d", m.BIAS_INDEX);
    //printf("\nINPUT_RANGE = %d - %d", m.INPUT_START, m.INPUT_END);
    //printf("\nHIDDEN_RANGE = %d - %d", m.HIDDEN_START, m.HIDDEN_END);
    //printf("\nOUTPUT_RANGE = %d - %d", m.OUTPUT_START, m.OUTPUT_END);

    //printf("\n\nINPUT_COUNT = %d", m.INPUT_COUNT);
    //printf("\nHIDDEN_COUNT = %d", m.HIDDEN_COUNT);
    //printf("\nOUTPUT_COUNT = %d", m.OUTPUT_COUNT);

    //printf("\n\nLENGTH = %d", m.LENGTH);

    for (unsigned int i = 0; i < m.LENGTH; i++) {
      //printf("\n\nNeuron %d, default func = %d", i, m.actFuncs.at(i));
      m.actFuncs.at(i) =
        (ActivationFuncEnum) (i % 3);

      for (unsigned int j = 0; j <= i; j++) {
        // printf("\nConnection %d = %d (%f) (default)",
        //       j,
        //       m.conns.at(m.LENGTH * i + j),
        //       m.weights.at(m.LENGTH * i + j));
        m.conns.at(m.LENGTH * i + j) = 1;
        m.weights.at(m.LENGTH * i + j) = ((float) j + 1) / ((float) m.LENGTH);
        //printf("\nConnection %d = %d (%f) (now)",
        //      j,
        //      m.conns.at(m.LENGTH * i + j),
        //      m.weights.at(m.LENGTH * i + j));
      }
    }

    m.setHiddenActivationFunction(TANH);
    m.setOutputActivationFunction(SOFTMAX);

    for (unsigned int i = m.OUTPUT_START + 1; i < m.OUTPUT_END; i += 2) {
      // Disable every second output neuron
      m.conns.at(m.LENGTH * i + i) = 0;
    }

    std::vector<double> outputs(m.OUTPUT_COUNT, 0);
    std::vector<double> inputs(m.INPUT_COUNT, 0);

    for (unsigned int i = m.INPUT_START; i < m.INPUT_END; i++) {
      inputs.at(i) = -i;
    }

    m.output(inputs.begin(), outputs.begin());

    double out_sum = 0;
    for (unsigned int i = 0; i < m.OUTPUT_COUNT; i++) {
      out_sum += outputs.at(i);
      printf("\nNetwork output.at(%d): %f", i, outputs.at(i));
    }
    printf("\nOutputsum = %f", out_sum);
    assert(abs(out_sum - 1.0) < 0.000001);
  }

  std::cout << "\nTestSoftmax done.";
}

void testLogRank() {
  std::cout << "\nTestLogRank...";

  // This test data is from
  // http://web.stanford.edu/~lutian/coursepdf/unitweek3.pdf Note that
  // the slide actually has a mistake when it denotes the value of E
  // in the example

  // 2 groups
  const unsigned int groupCount = 2;
  std::vector<unsigned int> groupCounts(groupCount, 0);
  // 6 in g0, 6 in g1
  groupCounts.at(0) = 6;
  groupCounts.at(1) = 6;

  const unsigned int length = 12;

  assert(groupCounts.at(0) + groupCounts.at(1) == length);

  std::vector<double> targets(2*length, 0);
  std::vector<unsigned int> groups(length, 0);

  // Define time, event and group status
  // Make sure it is time ordered
  int i = 0;
  groups.at(i) = 1;
  targets.at(2*i) = 3.1;
  targets.at(2*i + 1) = 1;

  i = 1;
  groups.at(i) = 1;
  targets.at(2*i) = 6.8;
  targets.at(2*i + 1) = 0;

  i = 2;
  groups.at(i) = 0;
  targets.at(2*i) = 8.7;
  targets.at(2*i + 1) = 1;

  i = 3;
  groups.at(i) = 0;
  targets.at(2*i) = 9.0;
  targets.at(2*i + 1) = 1;

  i = 4;
  groups.at(i) = 1;
  targets.at(2*i) = 9.0;
  targets.at(2*i + 1) = 1;

  i = 5;
  groups.at(i) = 1;
  targets.at(2*i) = 9.0;
  targets.at(2*i + 1) = 1;

  i = 6;
  groups.at(i) = 0;
  targets.at(2*i) = 10.1;
  targets.at(2*i + 1) = 0;

  i = 7;
  groups.at(i) = 1;
  targets.at(2*i) = 11.3;
  targets.at(2*i + 1) = 0;

  i = 8;
  groups.at(i) = 0;
  targets.at(2*i) = 12.1;
  targets.at(2*i + 1) = 0;

  i = 9;
  groups.at(i) = 1;
  targets.at(2*i) = 16.2;
  targets.at(2*i + 1) = 1;

  i = 10;
  groups.at(i) = 0;
  targets.at(2*i) = 18.7;
  targets.at(2*i + 1) = 1;

  i = 11;
  groups.at(i) = 0;
  targets.at(2*i) = 23.1;
  targets.at(2*i + 1) = 0;

  printf("\nLogRank:");
  double stat = TaroneWareMeanPairwise(targets, groups, groupCounts,
                                       length, TaroneWareType::LOGRANK);

  printf("\nStat = %f", stat);

  assert(abs(stat - 1.620508) < 0.000001);

  // Should be equal to high-low for two groups
  double stathighlow  = TaroneWareHighLow(targets, groups, groupCounts,
                                          length, groupCount,
                                          TaroneWareType::LOGRANK);

  printf("\nStatHighLow = %f", stathighlow);
  assert(stat == stathighlow);

  printf("\nGehan:");
  stat = TaroneWareMeanPairwise(targets, groups, groupCounts,
                                length,
                                TaroneWareType::GEHAN);

  printf("\nStat = %f", stat);

  stathighlow  = TaroneWareHighLow(targets, groups, groupCounts,
                                   length, groupCount,
                                   TaroneWareType::GEHAN);

  printf("\nStatHighLow = %f", stathighlow);
  assert(stat == stathighlow);

  printf("\nTaroneWare:");
  stat = TaroneWareMeanPairwise(targets, groups, groupCounts,
                                length,
                                TaroneWareType::TARONEWARE);

  printf("\nStat = %f", stat);

  stathighlow  = TaroneWareHighLow(targets, groups, groupCounts,
                                   length, groupCount,
                                   TaroneWareType::TARONEWARE);

  printf("\nStatHighLow = %f", stathighlow);
  assert(stat == stathighlow);


  std::cout << "\nTestLogRank done.";
}

void testSurvAreaNoCens() {
  std::cout << "\nTestSurvAreaNoCens...";

  // 2 groups
  const unsigned int groupCount = 2;
  std::vector<unsigned int> groupCounts(groupCount, 0);

  groupCounts.at(0) = 6;
  groupCounts.at(1) = 6;

  const unsigned int length = 12;

  assert(groupCounts.at(0) + groupCounts.at(1) == length);

  std::vector<double> targets(2*length, 0);
  std::vector<unsigned int> groups(length, 0);

  // Define time, event and group status
  // Make sure it is time ordered

  // First 6 are of group two, and have NEGATIVE TIMES
  // Should be ignored...
  int i = 0;
  groups.at(i) = 1;
  targets.at(2*i) = -6;
  targets.at(2*i + 1) = 1;

  i = 1;
  groups.at(i) = 1;
  targets.at(2*i) = -5;
  targets.at(2*i + 1) = 0;

  i = 2;
  groups.at(i) = 1;
  targets.at(2*i) = -4;
  targets.at(2*i + 1) = 1;

  i = 3;
  groups.at(i) = 1;
  targets.at(2*i) = -3;
  targets.at(2*i + 1) = 1;

  i = 4;
  groups.at(i) = 1;
  targets.at(2*i) = -2;
  targets.at(2*i + 1) = 1;

  i = 5;
  groups.at(i) = 1;
  targets.at(2*i) = -1;
  targets.at(2*i + 1) = 1;

  // Group 1 has normal times, one space apart
  i = 6;
  groups.at(i) = 0;
  targets.at(2*i) = 1;
  targets.at(2*i + 1) = 1;

  i = 7;
  groups.at(i) = 0;
  targets.at(2*i) = 2;
  targets.at(2*i + 1) = 1;

  i = 8;
  groups.at(i) = 0;
  targets.at(2*i) = 3;
  targets.at(2*i + 1) = 1;

  i = 9;
  groups.at(i) = 0;
  targets.at(2*i) = 4;
  targets.at(2*i + 1) = 1;

  i = 10;
  groups.at(i) = 0;
  targets.at(2*i) = 5;
  targets.at(2*i + 1) = 1;

  i = 11;
  groups.at(i) = 0;
  targets.at(2*i) = 6;
  targets.at(2*i + 1) = 1;

  assert(3.5 == SurvArea(targets, groups, groupCounts, length, false));

}


void testSurvAreaSameCens() {
  std::cout << "\nTestSurvAreaSameCens...";

  // 2 groups
  const unsigned int groupCount = 1;
  std::vector<unsigned int> groupCounts(groupCount, 0);

  groupCounts.at(0) = 6;

  const unsigned int length = 6;

  assert(groupCounts.at(0) == length);

  std::vector<double> targets(2*length, 0);
  std::vector<unsigned int> groups(length, 0);

  // Define time, event and group status
  // Make sure it is time ordered

  // Group 1 has normal times, one space apart
  int i = 0;
  groups.at(i) = 0;
  targets.at(2*i) = 1;
  targets.at(2*i + 1) = 1;

  i = 1;
  groups.at(i) = 0;
  targets.at(2*i) = 2;
  targets.at(2*i + 1) = 1;

  // This is censored at the same time as previous
  i = 2;
  groups.at(i) = 0;
  targets.at(2*i) = 2;
  targets.at(2*i + 1) = 0;

  i = 3;
  groups.at(i) = 0;
  targets.at(2*i) = 4;
  targets.at(2*i + 1) = 1;

  i = 4;
  groups.at(i) = 0;
  targets.at(2*i) = 5;
  targets.at(2*i + 1) = 1;

  i = 5;
  groups.at(i) = 0;
  targets.at(2*i) = 6;
  targets.at(2*i + 1) = 1;

  assert(abs(1.0 + 5.0/6.0 + 5.0/6.0*3.0/4.0*2.0 + 5.0/6.0*2.0/4.0 + 5.0/6.0/4.0 -
             SurvArea(targets, groups, groupCounts, length, false)) < 0.00000001);

}

void testSurvAreaMidCens() {
  std::cout << "\nTestSurvAreaMidCens...";

  // 2 groups
  const unsigned int groupCount = 1;
  std::vector<unsigned int> groupCounts(groupCount, 0);

  groupCounts.at(0) = 6;

  const unsigned int length = 6;

  assert(groupCounts.at(0) == length);

  std::vector<double> targets(2*length, 0);
  std::vector<unsigned int> groups(length, 0);

  // Define time, event and group status
  // Make sure it is time ordered

  // Group 1 has normal times, one space apart
  int i = 0;
  groups.at(i) = 0;
  targets.at(2*i) = 1;
  targets.at(2*i + 1) = 1;

  i = 1;
  groups.at(i) = 0;
  targets.at(2*i) = 2;
  targets.at(2*i + 1) = 1;

  // This is censored between two event times
  i = 2;
  groups.at(i) = 0;
  targets.at(2*i) = 3;
  targets.at(2*i + 1) = 0;

  i = 3;
  groups.at(i) = 0;
  targets.at(2*i) = 4;
  targets.at(2*i + 1) = 1;

  i = 4;
  groups.at(i) = 0;
  targets.at(2*i) = 5;
  targets.at(2*i + 1) = 1;

  i = 5;
  groups.at(i) = 0;
  targets.at(2*i) = 6;
  targets.at(2*i + 1) = 1;

  assert(abs(1.0 + 5.0/6.0 + 4.0/6.0*2.0 + 4.0/9.0 + 2.0/9.0 -
             SurvArea(targets, groups, groupCounts, length, false)) < 0.00000001);
}


void testSurvAreaEndCens1() {
  std::cout << "\nTestSurvAreaEndCens1...";

  // 2 groups
  const unsigned int groupCount = 1;
  std::vector<unsigned int> groupCounts(groupCount, 0);

  groupCounts.at(0) = 6;

  const unsigned int length = 6;

  assert(groupCounts.at(0) == length);

  std::vector<double> targets(2*length, 0);
  std::vector<unsigned int> groups(length, 0);

  // Define time, event and group status
  // Make sure it is time ordered

  // Group 1 has normal times, one space apart
  int i = 0;
  groups.at(i) = 0;
  targets.at(2*i) = 1;
  targets.at(2*i + 1) = 1;

  i = 1;
  groups.at(i) = 0;
  targets.at(2*i) = 2;
  targets.at(2*i + 1) = 1;

  i = 2;
  groups.at(i) = 0;
  targets.at(2*i) = 3;
  targets.at(2*i + 1) = 1;

  i = 3;
  groups.at(i) = 0;
  targets.at(2*i) = 4;
  targets.at(2*i + 1) = 1;

  i = 4;
  groups.at(i) = 0;
  targets.at(2*i) = 5;
  targets.at(2*i + 1) = 1;

  // Censored at the end, same time as last event
  i = 5;
  groups.at(i) = 0;
  targets.at(2*i) = 5;
  targets.at(2*i + 1) = 0;

  assert(abs(1.0 + 5.0/6.0 + 4.0/6.0 + 3.0/6.0 + 2.0/6.0 -
             SurvArea(targets, groups, groupCounts, length, false)) < 0.00000001);
}


void testSurvAreaEndCens2() {
  std::cout << "\nTestSurvAreaEndCens2...";

  // 2 groups
  const unsigned int groupCount = 1;
  std::vector<unsigned int> groupCounts(groupCount, 0);

  groupCounts.at(0) = 6;

  const unsigned int length = 6;

  assert(groupCounts.at(0) == length);

  std::vector<double> targets(2*length, 0);
  std::vector<unsigned int> groups(length, 0);

  // Define time, event and group status
  // Make sure it is time ordered

  // Group 1 has normal times, one space apart
  int i = 0;
  groups.at(i) = 0;
  targets.at(2*i) = 1;
  targets.at(2*i + 1) = 1;

  i = 1;
  groups.at(i) = 0;
  targets.at(2*i) = 2;
  targets.at(2*i + 1) = 1;

  i = 2;
  groups.at(i) = 0;
  targets.at(2*i) = 3;
  targets.at(2*i + 1) = 1;

  i = 3;
  groups.at(i) = 0;
  targets.at(2*i) = 4;
  targets.at(2*i + 1) = 1;

  i = 4;
  groups.at(i) = 0;
  targets.at(2*i) = 5;
  targets.at(2*i + 1) = 1;

  // Censored at the end, after last event
  i = 5;
  groups.at(i) = 0;
  targets.at(2*i) = 6;
  targets.at(2*i + 1) = 0;

  assert(abs(1.0 + 5.0/6.0 + 4.0/6.0 + 3.0/6.0 + 2.0/6.0 + 1.0/6.0 -
             SurvArea(targets, groups, groupCounts, length, false)) < 0.00000001);
}

int main( int argc, const char* argv[] )
{
  geneticTest1();
  geneticXOR();
  rproptest();

  geneticXORDroput();
  rpropdropouttest();


  testSurvAreaNoCens();
  testSurvAreaSameCens();
  testSurvAreaMidCens();
  testSurvAreaEndCens1();
  testSurvAreaEndCens2();

  testLogRank();
  testSoftmax();
  survLikTests();
  survMSETests();
  errorTests();
  derivTests();
  matrixtest();
  randomtest();
  rpropalloctest();
  //survcachealloctest();
  testSurvCache();
  rpropsurvlik();
  printf("\nEND OF TEST\n");
}
