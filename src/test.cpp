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
      //printf("\n\nNeuron %d, default func = %d", i, m.actFuncs[i]);
      m.actFuncs[i] =
        (ActivationFuncEnum) (i % 3);

      for (unsigned int j = 0; j < m.LENGTH; j++) {
        // printf("\nConnection %d = %d (%f) (default)",
        //       j,
        //       m.conns[m.LENGTH * i + j],
        //       m.weights[m.LENGTH * i + j]);
        m.conns[m.LENGTH * i + j] = 1;
        m.weights[m.LENGTH * i + j] = j + 1;
        //printf("\nConnection %d = %d (%f) (now)",
        //      j,
        //      m.conns[m.LENGTH * i + j],
        //      m.weights[m.LENGTH * i + j]);
      }
    }

    m.setHiddenActivationFunction(TANH);
    m.setOutputActivationFunction(LINEAR);

    // printf("\n\nTANH = %d, LOGSIG = %d\n", TANH, LOGSIG);
    for (unsigned int i = m.HIDDEN_START; i < m.HIDDEN_END; i++) {
      // printf("\nHIDDEN(%d).actFunc = %d", i, m.actFuncs[i]);
    }
    // printf("\n\nTANH = %d, LOGSIG = %d\n", TANH, LOGSIG);
    for (unsigned int i = m.OUTPUT_START; i < m.OUTPUT_END; i++) {
      // printf("\nOUTPUT(%d).actFunc = %d", i, m.actFuncs[i]);
    }

    double *outputs = new double[m.OUTPUT_COUNT]();
    double *inputs = new double[m.INPUT_COUNT]();

    for (unsigned int i = m.INPUT_START; i < m.INPUT_END; i++) {
      inputs[i] = 2;
    }

    m.output(inputs, outputs);

    for (unsigned int i = 0; i < m.OUTPUT_COUNT; i++) {
      // printf("\nNetwork output[%d]: %f", i, outputs[i]);
    }

    delete[] outputs;
    delete[] inputs;
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

  GeneticNetwork net1(5, 3, 1);
  GeneticNetwork net2(net1.INPUT_COUNT,
                      net1.HIDDEN_COUNT,
                      net1.OUTPUT_COUNT);

  // printf("\n5 = %d\n3 = %d\n1 = %d",
//         net2.INPUT_COUNT,
//         net2.HIDDEN_COUNT,
 //        net2.OUTPUT_COUNT);

  Random rand;
  GeneticMutator mutator(rand);

  mutator.randomizeNetwork(net1, 1.0);

  mutator.randomizeNetwork(net2, 1.0);

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
  net2.cloneNetwork(net1);

  // printf("\nWeight diff: %f vs %f
//\nConn diff: %d vs %d
//\nActF diff: %d vs %d",
//         net1.weights[9],
  //       net2.weights[9],
    //     net1.conns[9],
    //     net2.conns[9],
     //    net1.actFuncs[3],
      //   net2.actFuncs[3]);


  printf("\nGeneticTest1 Done.");
}

void geneticXOR() {
  std::cout << "\nGeneticXOR...";
  GeneticNetwork net(2, 5, 1);
  net.setGenerations(10);
  net.setWeightMutationChance(0.5);
  net.setWeightMutationFactor(0.3);
  net.connsMutationChance = 0.5;
  net.actFuncMutationChance = 0.5;
  net.setCrossoverChance(0.6);

  // define inputs
  double X[2*4]{0,0,
      0,1,
      1,0,
      1,1};
  // printf("\nbah %f", X[7]);

  // define targets
  double Y[4]{0, 1, 1, 0};
  // printf("\nbah %f", Y[2]);

  net.learn(X, Y, 4);

  double preds[1];


   std::cout << "\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X + 2 * i, preds);
     std::cout << X[2*i] << " "<< X[2*i + 1]
              << " : " << preds[0] << "\n";
  }

  // Print structure
  // std::cout << "\n\nWeights";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    // std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      // std::cout << " " << net.weights[j + i*net.LENGTH];
    }
  }

  // std::cout << "\n\nConss";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    // std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      // std::cout << " " << net.conns[j + i*net.LENGTH];
    }
  }

  // std::cout << "\n\nActFuncs";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    // std::cout << "\nN" << i << ": " << net.actFuncs[i];
  }

  std::cout << "\nGeneticXOR Done.";
}

void rpropsurvlik() {
  std::cout << "\nRPropSurvLikTest...";

  Random r;
  RPropNetwork net(2, 8, 2);

  // Set up a feedforward structure
  for (unsigned int i = net.OUTPUT_START; i < net.LENGTH; i++) {
    for (unsigned int j = net.HIDDEN_START; j < net.HIDDEN_END; j++) {
      net.conns[net.LENGTH * i + j] = 1;
      net.weights[net.LENGTH * i + j] = r.normal();
    }
    // Also activate self
    net.conns[net.LENGTH * i + i] = 1;
  }
  for (unsigned int i = net.HIDDEN_START; i < net.HIDDEN_END; i++) {
    for (unsigned int j = 0; j < net.BIAS_END; j++) {
      net.conns[net.LENGTH * i + j] = 1;
      net.weights[net.LENGTH * i + j] = r.normal();
    }
    // Also activate self
    net.conns[net.LENGTH * i + i] = 1;
  }
  net.setHiddenActivationFunction(TANH);
  net.setOutputActivationFunction(LINEAR);

  double X[2*8]{0,0,
      0,1,
      1,0,
      1,1,
      0,0,
      0,1,
      1,0,
      1,1};

  // define targets
  // initial censored point which segfaulted at some time
  double Y[2*8]{0, 0,
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

  double preds[2];
  // std::cout << "\n\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X + 2 * i, preds);
    // std::cout << X[2*i] << " "<< X[2*i + 1]
       //       << " : " << std::round(preds[0])
        //      << " (" << Y[i] << ")"<< "\n";
  }

  std::cout << "\nRPropSurvLikTest Done.";
}

void rproptest() {
  std::cout << "\nRPropTest...";

  Random r;
  RPropNetwork net(2, 8, 1);

  // Set up a feedforward structure
  for (unsigned int i = net.OUTPUT_START; i < net.LENGTH; i++) {
    for (unsigned int j = net.HIDDEN_START; j < net.HIDDEN_END; j++) {
      net.conns[net.LENGTH * i + j] = 1;
      net.weights[net.LENGTH * i + j] = r.normal();
    }
    // Also activate self
    net.conns[net.LENGTH * i + i] = 1;
  }
  for (unsigned int i = net.HIDDEN_START; i < net.HIDDEN_END; i++) {
    for (unsigned int j = 0; j < net.BIAS_END; j++) {
      net.conns[net.LENGTH * i + j] = 1;
      net.weights[net.LENGTH * i + j] = r.normal();
    }
    // Also activate self
    net.conns[net.LENGTH * i + i] = 1;
  }
  net.setHiddenActivationFunction(TANH);
  net.setOutputActivationFunction(LOGSIG);

  // xor
  // define inputs
  double X[2*4]{0,0,
      0,1,
      1,0,
      1,1};

  // define targets
  double Y[4]{0, 1, 1, 0};

  // Print structure
  std::cout << "\n\nConns before";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j <= i; j++) {
      std::cout << " " << net.conns[j + i*net.LENGTH];
    }
  }

  net.setMaxEpochs(10000);
  net.setMaxError(0.0);
  net.setMinErrorFrac(0.01);
  if ( 0 < net.learn(X, Y, 4)) {
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

  double preds[1];
  // std::cout << "\n\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X + 2 * i, preds);

    std::cout << "\n  " << X[2*i] << " "<< X[2*i + 1]
              << " : " << preds[0]
              << " (" << Y[i] << ")";

    double diff = preds[0] - Y[i];
    if (diff < 0) diff = -diff;

    assert(diff < 0.1);
  }

  std::cout << "\nRPropTest Done.";
}

void rpropalloctest() {
  std::cout << "\nRPropAllocTest...";

  Random r;
  RPropNetwork net(2, 8, 2);

  // Set up a feedforward structure
  for (unsigned int i = net.OUTPUT_START; i < net.LENGTH; i++) {
    for (unsigned int j = net.HIDDEN_START; j < net.HIDDEN_END; j++) {
      net.conns[net.LENGTH * i + j] = 1;
      net.weights[net.LENGTH * i + j] = r.normal();
    }
  }
  for (unsigned int i = net.HIDDEN_START; i < net.HIDDEN_END; i++) {
    for (unsigned int j = 0; j < net.BIAS_END; j++) {
      net.conns[net.LENGTH * i + j] = 1;
      net.weights[net.LENGTH * i + j] = r.normal();
    }
  }
  net.setHiddenActivationFunction(TANH);
  net.setOutputActivationFunction(LOGSIG);

  // xor
  // define inputs
  unsigned int limit = 100000;

  double *X = new double[2*limit]();
  // define targets
  double *Y = new double[2*limit]();

  // Not interested in training result

  net.setMaxEpochs(1);
  net.setMaxError(0.001);

  net.setErrorFunction(ErrorFunction::ERROR_MSE);
  if ( 0 < net.learn(X, Y, limit)) {
    throw "Shit hit the fan";
  }

  delete[] X;
  delete[] Y;

  std::cout << "\nRPropAllocTest Done.";
}

// Takes a few minutes to run
void survcachealloctest() {
  std::cout << "\nSurvCacheAllocTest...";

  unsigned int limit = 100000;
  double *Y = new double[2*limit]();

  SurvErrorCache *cache = new SurvErrorCache();

  cache->verifyInit(Y, limit);

  delete[] Y;
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
  double targets[] = {
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

      double outputs[rows*cols];
      double targets[rows*cols];
      double errors[rows*cols];

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

      double outputs[rows*cols];
      double targets[rows*cols];
      double derivs[rows*cols];

      // init
      for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {
          outputs[i * cols + j] = x;
          targets[i * cols + j] = y;
        }
      }

      for (unsigned int i = 0; i < rows; i++) {
        unsigned int index = i * cols;
        getDerivative(ErrorFunction::ERROR_MSE,
                      targets, rows, cols,
                      outputs, index, derivs + index);
        for (unsigned int j = 0; j < cols; j++) {
          //std::cout << i << "," << j << " d: " << derivs[index + j] << "\n";;
          assertSame(derivs[index + j], (x-y));
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

  double outputs[rows*cols];
  double targets[rows*cols];
  double errors[rows*cols];

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

  double outputs[rows*cols];
  double targets[rows*cols];
  double errors[rows*cols];

  //////////////
  // First I want to test uncensored points
  // Then we go on to censored first
  // censored middle
  // censored last
  //////////////

  for (i = 0; i < rows; i++)
  {
      // Targets
      targets[i * cols] = i;
      targets[i * cols + 1] = 1.0;
      // Even indices underestimate
      if (i % 2 == 0)
      {
        outputs[i * cols] = i - 2.0;
      }
      else // Odd overestimate
      {
        outputs[i * cols] = i + 2.0;
      }
  }

  // Censor first point
  targets[0 + 1] = 0;

  // Censor two in the middle
  targets[10 * cols + 1] = 0;
  targets[11 * cols + 1] = 0;

  // Censor last two
  targets[(rows - 2) * cols + 1] = 0;
  targets[(rows - 1) * cols + 1] = 0;

  getAllErrors(ErrorFunction::ERROR_SURV_LIKELIHOOD,
               targets, rows, cols,
               outputs, errors);

  for (i = 0; i < rows; i++) {
    printf("\n  E: %f", errors[i * cols]);
      //std::cout << "\ne: " << errors[i * cols] << "\n";
    //assertSame(errors[i * cols], 4.0);
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
      //printf("\n\nNeuron %d, default func = %d", i, m.actFuncs[i]);
      m.actFuncs[i] =
        (ActivationFuncEnum) (i % 3);

      for (unsigned int j = 0; j <= i; j++) {
        // printf("\nConnection %d = %d (%f) (default)",
        //       j,
        //       m.conns[m.LENGTH * i + j],
        //       m.weights[m.LENGTH * i + j]);
        m.conns[m.LENGTH * i + j] = 1;
        m.weights[m.LENGTH * i + j] = ((float) j + 1) / ((float) m.LENGTH);
        //printf("\nConnection %d = %d (%f) (now)",
        //      j,
        //      m.conns[m.LENGTH * i + j],
        //      m.weights[m.LENGTH * i + j]);
      }
    }

    m.setHiddenActivationFunction(TANH);
    m.setOutputActivationFunction(SOFTMAX);

    // printf("\n\nTANH = %d, LOGSIG = %d\n", TANH, LOGSIG);
    for (unsigned int i = m.HIDDEN_START; i < m.HIDDEN_END; i++) {
      // printf("\nHIDDEN(%d).actFunc = %d", i, m.actFuncs[i]);
    }
    // printf("\n\nTANH = %d, LOGSIG = %d\n", TANH, LOGSIG);
    for (unsigned int i = m.OUTPUT_START + 1; i < m.OUTPUT_END; i += 2) {
      // Disable every second output neuron
      m.conns[m.LENGTH * i + i] = 0;
    }

    double *outputs = new double[m.OUTPUT_COUNT]();
    double *inputs = new double[m.INPUT_COUNT]();

    for (unsigned int i = m.INPUT_START; i < m.INPUT_END; i++) {
      inputs[i] = -99;
    }

    m.output(inputs, outputs);

    double out_sum = 0;
    for (unsigned int i = 0; i < m.OUTPUT_COUNT; i++) {
      out_sum += outputs[i];
      printf("\nNetwork output[%d]: %f", i, outputs[i]);
    }
    printf("\nOutputsum = %f", out_sum);
    assert(abs(out_sum - 1.0) < 0.000001);

    delete[] outputs;
    delete[] inputs;
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
  unsigned int *groupCounts = new unsigned int[groupCount];
  // 5 in g0, 6 in g1
  groupCounts[0] = 6;
  groupCounts[1] = 6;

  // 11 times total
  const unsigned int length = 12;

  assert(groupCounts[0] + groupCounts[1] == length);

  double *targets = new double[2*length];
  unsigned int *groups = new unsigned int[length];

  // Define time, event and group status
  // Make sure it is time ordered
  int i = 0;
  groups[i] = 1;
  targets[2*i] = 3.1;
  targets[2*i + 1] = 1;

  i = 1;
  groups[i] = 1;
  targets[2*i] = 6.8;
  targets[2*i + 1] = 0;

  i = 2;
  groups[i] = 0;
  targets[2*i] = 8.7;
  targets[2*i + 1] = 1;

  i = 3;
  groups[i] = 0;
  targets[2*i] = 9.0;
  targets[2*i + 1] = 1;

  i = 4;
  groups[i] = 1;
  targets[2*i] = 9.0;
  targets[2*i + 1] = 1;

  i = 5;
  groups[i] = 1;
  targets[2*i] = 9.0;
  targets[2*i + 1] = 1;

  i = 6;
  groups[i] = 0;
  targets[2*i] = 10.1;
  targets[2*i + 1] = 0;

  i = 7;
  groups[i] = 1;
  targets[2*i] = 11.3;
  targets[2*i + 1] = 0;

  i = 8;
  groups[i] = 0;
  targets[2*i] = 12.1;
  targets[2*i + 1] = 0;

  i = 9;
  groups[i] = 1;
  targets[2*i] = 16.2;
  targets[2*i + 1] = 1;

  i = 10;
  groups[i] = 0;
  targets[2*i] = 18.7;
  targets[2*i + 1] = 1;

  i = 11;
  groups[i] = 0;
  targets[2*i] = 23.1;
  targets[2*i + 1] = 0;

  double stat = TaroneWareStatistic(targets, groups, groupCounts,
                                    length, groupCount);

  printf("\nStat = %f", stat);
  printf("\nsqrt(Stat) = %f", sqrt(stat));

  assert(abs(stat - 1.620508) < 0.000001);

  // Cleanup
  delete[] targets;
  delete[] groupCounts;
  delete[] groups;

  std::cout << "\nTestLogRank done.";
}

int main( int argc, const char* argv[] )
{
  testLogRank();
  testSoftmax();
  survLikTests();
  survMSETests();
  errorTests();
  derivTests();
  matrixtest();
  randomtest();
  //rpropalloctest();
  //survcachealloctest();
  //geneticTest1();
  geneticXOR();
  rproptest();
  testSurvCache();
  rpropsurvlik();
  printf("\nEND OF TEST\n");
}
