#include "MatrixNetwork.h"
#include "activationfunctions.h"
#include <stdio.h>
#include "Random.h"

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
  Random r;

  printf("\n\nRandom numbers:");
  printf("\nUniform: %f", r.uniform());
  printf("\nNormal: %f", r.normal());
  printf("\nGeometric(10): %d", r.geometric(10));

}

int main( int argc, const char* argv[] )
{
  matrixtest();
  randomtest();
  printf("\nEND OF TEST\n");
}
