/*
  A network which trains by use of the cascade correlation algorithm.
  The output nodes are trained by a genetic algorithm.
  The hidden nodes are trained with RPROP to maximize correlation.
*/
#ifndef _CASCADENETWORK_H_
#define _CASCADENETWORK_H_

#include "RPropNetwork.h"
#include "FFNeuron.h"
#include "FFNetwork.h"
#include <vector>

/*
  Neuron which is trained by RProp.
  Used in hidden layers.
*/
class RCascadeNeuron : public RPropNeuron {
public:
  RCascadeNeuron(int id);
  RCascadeNeuron(int id, double (*activationFunction)(double),
                 double (*activationDerivative)(double));
  virtual ~RCascadeNeuron();

  virtual void learn(double *patError, double *error, double *X,
                       double *outputs, unsigned int rows,
                       unsigned int numOfInputs);
  virtual void applyWeightUpdates(int covariance);

virtual void addLocalError(double error);
virtual void calcLocalDerivative(double *inputs);
};

class CascadeNetwork : public RPropNetwork {
 protected:
  std::vector<RCascadeNeuron*> *hiddenRCascadeNeurons;
  unsigned int maxHidden;
  unsigned int maxHiddenEpochs;

 public:
  CascadeNetwork(unsigned int numOfInputs, unsigned int numOfOutput);
  // delete hiddenRCascadeNeurons!
  virtual ~CascadeNetwork();
  virtual void initNodes();
  /*
   * X is an array of input arrays.
   * Y is an array of target outputs. total length is 'rows * numOfInputs'
   * and 'rows * numOfOutputs'
   */
  virtual void learn(double *X, double *Y, unsigned int rows);

  virtual unsigned int getNumOfHidden() const;
  virtual double *output(double *inputs, double *output);

  virtual void trainOutputs(double *X, double *Y, unsigned int rows);

  virtual void calcErrors(double *X, double *Y, unsigned int rows,
                            double *patError, double *error, double *outputs);

  virtual Neuron* getHiddenNeuron(unsigned int id) const;
  virtual bool getNeuronWeightFromHidden(unsigned int fromId, int toId, double *weight);
  virtual bool getInputWeightFromHidden(unsigned int fromId, unsigned int toIndex, double *weight);

  virtual unsigned int getMaxHidden() const;
  virtual void setMaxHidden(unsigned int num);
  virtual unsigned int getMaxHiddenEpochs() const;
  virtual void setMaxHiddenEpochs(unsigned int num);

};

#endif /* _CASCADENETWORK_H_ */
