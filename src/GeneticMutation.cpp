#include "GeneticMutation.hpp"
#include "MatrixNetwork.hpp"
#include "activationfunctions.hpp"
#include <stdio.h>

GeneticMutator::GeneticMutator(Random &rand) :
  rand(rand)
{}

GeneticMutator::~GeneticMutator()
{}

void GeneticMutator::randomizeNetwork(MatrixNetwork &net,
                                      const double weightStdDev)
{
  mutateWeights(net, 1.0, weightStdDev);
  mutateConns(net, 1.0);
  mutateActFuncs(net, 1.0);
}

void GeneticMutator::mutateWeights(MatrixNetwork &net,
                                   const double chance,
                                   const double stddev)
{
  unsigned int i;
  double roll;

  for (i = 0; i < net.LENGTH * net.LENGTH; i++) {
    roll = this->rand.uniform();
    if (roll < chance) {
      // Mutate the weight
      net.weights.at(i) += this->rand.normal() * stddev;
    }
  }
}

void GeneticMutator::mutateConns(MatrixNetwork &net,
                                 const double chance)
{
  unsigned int i, bit;
  double roll;

  for (i = 0; i < net.LENGTH * net.LENGTH; i++) {
    roll = this->rand.uniform();
    if (roll < chance) {
      bit = this->rand.randBit();
      net.conns.at(i) = bit;
    }
  }
}

void GeneticMutator::mutateActFuncs(MatrixNetwork &net,
                                    const double chance)
{
  unsigned int i, choice;
  double roll;

  for (i = 0; i < net.LENGTH; i++) {
    roll = this->rand.uniform();
    if (roll < chance) {
      // Select new function
      choice = this->rand.uniformNumber(0, 3);
      // Select a function OTHER than the current
      switch(choice) {
      case 0:
        if (LINEAR == net.actFuncs.at(i)) {
          net.actFuncs.at(i) = LOGSIG;
        }
        else {
          net.actFuncs.at(i) = LINEAR;
        }
        break;
      case 1:
        if (LOGSIG == net.actFuncs.at(i)) {
          net.actFuncs.at(i) = TANH;
        }
        else {
          net.actFuncs.at(i) = LOGSIG;
        }
        break;
      case 2:
      default:
        if (TANH == net.actFuncs.at(i)) {
          net.actFuncs.at(i) = LINEAR;
        }
        else {
          net.actFuncs.at(i) = TANH;
        }
        break;
      }
    }
  }
}
