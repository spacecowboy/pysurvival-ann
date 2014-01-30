#include "MatrixNetwork.hpp"
#include "activationfunctions.hpp"
#include <stdio.h>
#include <limits> // max int
#include "Random.hpp"
#include "global.hpp"
#include "GeneticNetwork.hpp"
#include "GeneticMutation.hpp"
#include "RPropNetwork.hpp"
#include <iostream>
#include <cmath>

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

void lockTest() {
  std::cout << "\nLockTest...";
    // printf("\n\nLocking mutex...");
    JGN_lockPopulation();
    // printf("\nMutex locked");

    // printf("\nUnlocking mutex...");
    JGN_unlockPopulation();
    // printf("\nMutex unlocked");
    std::cout << "\nLockTest Done.";
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
  net.setGenerations(100);
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


  // std::cout << "\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X + 2 * i, preds);
    // std::cout << X[2*i] << " "<< X[2*i + 1]
      //        << " : " << preds[0] << "\n";
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
  }
  for (unsigned int i = net.HIDDEN_START; i < net.HIDDEN_END; i++) {
    for (unsigned int j = 0; j < net.BIAS_END; j++) {
      net.conns[net.LENGTH * i + j] = 1;
      net.weights[net.LENGTH * i + j] = r.normal();
    }
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

  double preds[1];
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
  }
  for (unsigned int i = net.HIDDEN_START; i < net.HIDDEN_END; i++) {
    for (unsigned int j = 0; j < net.BIAS_END; j++) {
      net.conns[net.LENGTH * i + j] = 1;
      net.weights[net.LENGTH * i + j] = r.normal();
    }
  }
  net.setHiddenActivationFunction(TANH);
  net.setOutputActivationFunction(LOGSIG);

  // define inputs
  double X[2*4]{0,0,
      0,1,
      1,0,
      1,1};

  // define targets
  double Y[4]{0, 1, 1, 0};

  // Print structure
  // std::cout << "\n\nWeights before";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    // std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      // std::cout << " " << net.weights[j + i*net.LENGTH];
    }
  }

  net.setMaxEpochs(10000);
  net.setMaxError(0.001);
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
    // std::cout << X[2*i] << " "<< X[2*i + 1]
       //       << " : " << std::round(preds[0])
        //      << " (" << Y[i] << ")"<< "\n";
  }

  std::cout << "\nRPropTest Done.";
}

void test_survcache() {
  // First method should be get prob, this is used in the rest
  // This should be a vector with length equal to entire times vector.
  // Probs at censored events is zero
  // Cumulative sum must > 0 and <= 1

  // Prob after method is just 1.0 - sum(Probs)


}


int main( int argc, const char* argv[] )
{
  matrixtest();
  randomtest();
  lockTest();
  //geneticTest1();
  //geneticXOR();
  rproptest();
  rpropsurvlik();
  printf("\nEND OF TEST\n");
}
