#include "MatrixNetwork.hpp"
#include "activationfunctions.hpp"
#include <stdio.h>
//#include "Random.hpp"
#include "global.hpp"
#include "GeneticNetwork.hpp"
#include "GeneticMutation.hpp"

void matrixtest() {
  	printf( "\nStarting matrix test...\n\n" );

    printf("\nConstructing 2, 2, 2");
    MatrixNetwork m(2, 0, 2);

    printf("\nBIAS_INDEX = %d", m.BIAS_INDEX);
    printf("\nINPUT_RANGE = %d - %d", m.INPUT_START, m.INPUT_END);
    printf("\nHIDDEN_RANGE = %d - %d", m.HIDDEN_START, m.HIDDEN_END);
    printf("\nOUTPUT_RANGE = %d - %d", m.OUTPUT_START, m.OUTPUT_END);

    printf("\n\nINPUT_COUNT = %d", m.INPUT_COUNT);
    printf("\nHIDDEN_COUNT = %d", m.HIDDEN_COUNT);
    printf("\nOUTPUT_COUNT = %d", m.OUTPUT_COUNT);

    printf("\n\nLENGTH = %d", m.LENGTH);

    for (unsigned int i = 0; i < m.LENGTH; i++) {
      printf("\n\nNeuron %d, default func = %d", i, m.actFuncs[i]);
      m.actFuncs[i] =
        (ActivationFuncEnum) (i % 3);

      for (unsigned int j = 0; j < m.LENGTH; j++) {
        printf("\nConnection %d = %d (%f) (default)",
               j,
               m.conns[m.LENGTH * i + j],
               m.weights[m.LENGTH * i + j]);
        m.conns[m.LENGTH * i + j] = 1;
        m.weights[m.LENGTH * i + j] = j + 1;
        printf("\nConnection %d = %d (%f) (now)",
               j,
               m.conns[m.LENGTH * i + j],
               m.weights[m.LENGTH * i + j]);
      }
    }

    m.setHiddenActivationFunction(TANH);
    m.setOutputActivationFunction(LINEAR);

    printf("\n\nTANH = %d, LOGSIG = %d\n", TANH, LOGSIG);
    for (unsigned int i = m.HIDDEN_START; i < m.HIDDEN_END; i++) {
      printf("\nHIDDEN(%d).actFunc = %d", i, m.actFuncs[i]);
    }
    printf("\n\nTANH = %d, LOGSIG = %d\n", TANH, LOGSIG);
    for (unsigned int i = m.OUTPUT_START; i < m.OUTPUT_END; i++) {
      printf("\nOUTPUT(%d).actFunc = %d", i, m.actFuncs[i]);
    }

    double *outputs = new double[m.OUTPUT_COUNT]();
    double *inputs = new double[m.INPUT_COUNT]();

    for (unsigned int i = m.INPUT_START; i < m.INPUT_END; i++) {
      inputs[i] = 2;
    }

    m.output(inputs, outputs);

    for (unsigned int i = 0; i < m.OUTPUT_COUNT; i++) {
      printf("\nNetwork output[%d]: %f", i, outputs[i]);
    }

	printf( "\n\nEnding matrix test...\n\n" );
    delete[] outputs;
    delete[] inputs;
}

void randomtest() {
    //Random r;

  printf("\n\nRandom numbers:");
  printf("\nUniform: %f", JGN_rand.uniform());
  printf("\nNormal: %f", JGN_rand.normal());
  printf("\nGeometric(10): %d", JGN_rand.geometric(10));

  printf("\nUniform number: %d", JGN_rand.uniformNumber(1, 10));

}

void lockTest() {
    printf("\n\nLocking mutex...");
    JGN_lockPopulation();
    printf("\nMutex locked");

    printf("\nUnlocking mutex...");
    JGN_unlockPopulation();
    printf("\nMutex unlocked");
}

void geneticTest1() {
  printf("\n\nCreating genetic networks...");

  GeneticNetwork net1(5, 3, 1);
  GeneticNetwork net2(net1.INPUT_COUNT,
                      net1.HIDDEN_COUNT,
                      net1.OUTPUT_COUNT);

  printf("\n5 = %d\n3 = %d\n1 = %d",
         net2.INPUT_COUNT,
         net2.HIDDEN_COUNT,
         net2.OUTPUT_COUNT);

  randomizeNetwork(net1, 1.0);

  randomizeNetwork(net2, 1.0);

  printf("\nWeight diff: %f vs %f \
\nConn diff: %d vs %d\
\nActF diff: %d vs %d",
         net1.weights[9],
         net2.weights[9],
         net1.conns[9],
         net2.conns[9],
         net1.actFuncs[3],
         net2.actFuncs[3]);

  printf("\nCloning...");
  net2.cloneNetwork(net1);

  printf("\nWeight diff: %f vs %f \
\nConn diff: %d vs %d\
\nActF diff: %d vs %d",
         net1.weights[9],
         net2.weights[9],
         net1.conns[9],
         net2.conns[9],
         net1.actFuncs[3],
         net2.actFuncs[3]);


  printf("\nGenetic test 1 done.");
}

void geneticXOR() {
  GeneticNetwork net(2, 8, 1);
  net.setGenerations(10);
  net.setWeightMutationChance(0.25);
  net.setWeightMutationFactor(0.3);
  net.connsMutationChance = 0.35;
  net.actFuncMutationChance = 0.35;
  net.setCrossoverChance(0.6);

  // define inputs
  double X[2*4]{0,0,
      0,1,
      1,0,
      1,1};
  printf("\nbah %f", X[7]);

  // define targets
  double Y[4]{0, 1, 1, 0};
  printf("\nbah %f", Y[2]);

  net.learn(X, Y, 4);

  double preds[1];


  std::cout << "\nPredictions\n";
  for (int i = 0; i < 4; i++) {
    net.output(X + 2 * i, preds);
    std::cout << X[2*i] << " "<< X[2*i + 1]
              << " : " << preds[0] << "\n";
  }

  // Print structure
  std::cout << "\n\nWeights";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      std::cout << " " << net.weights[j + i*net.LENGTH];
    }
  }

  std::cout << "\n\nConss";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    std::cout << "\nN" << i << ":";
    for (unsigned int j = 0; j < i; j++) {
      std::cout << " " << net.conns[j + i*net.LENGTH];
    }
  }

  std::cout << "\n\nActFuncs";
  for (unsigned int i = net.HIDDEN_START; i < net.OUTPUT_END; i++) {
    std::cout << "\nN" << i << ": " << net.actFuncs[i];
  }

}

int main( int argc, const char* argv[] )
{
  matrixtest();
  randomtest();
  lockTest();
  geneticTest1();
  geneticXOR();
  printf("\nEND OF TEST\n");
}
