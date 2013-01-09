/*
  A network which trains by use of the cascade correlation algorithm.
  The output nodes are trained by a genetic algorithm.
  The hidden nodes are trained with RPROP to maximize correlation.
*/
#ifndef _COXCASCADENETWORK_H_
#define _COXCASCADENETWORK_H_

#include "FFNeuron.h"
#include "CascadeNetwork.h"
//#include "FFNetwork.h"
#include <vector>
#include <string>
#include <RInside.h> // for embedded R via RInside

//static std::string getCoxCmd(std::vector<std::string> *colNames);

class CoxNeuron : public Neuron {
 protected:
  RInside *R;

  unsigned int getCoxNumOfCols();

  void convertToRColumns(double *X, double *Y, unsigned int rows,
                           Rcpp::CharacterVector *colNames,
                           std::vector<std::vector<double>*> *listOfColumns);


public:
  CoxNeuron(int id);
  virtual ~CoxNeuron();

  /*
	 * Calculate the output of this neuron.
	 * Performs predict(cox, ....)
     * Assumes connected nodes have already computed their outputs.
	 */
  virtual double output(double *inputs);
  double output(double *inputs, unsigned int rows, double *outputs);

  /*
   * Fits a cox model to the data and saves the model for later
   */
  void fit(double *X, double *Y, unsigned int rows);

  double getConcordance();
  //virtual std::string getSummary();
};

class CoxCascadeNetwork : public CascadeNetwork {
 public:
  CoxCascadeNetwork(unsigned int numOfInputs, unsigned int numOfOutput);
  virtual ~CoxCascadeNetwork();
  virtual void initNodes();

  /*
   * X is an array of input arrays.
   * Y is an array of target outputs. total length is 'rows * numOfInputs'
   * and 'rows * numOfOutputs'
   */
  //virtual void learn(double *X, double *Y, unsigned int rows);

  //virtual double *output(double *inputs, double *output);

  virtual void trainOutputs(double *X, double *Y, unsigned int rows);

  virtual void calcErrors(double *X, double *Y, unsigned int rows,
                            double *patError, double *error, double *outputs);
};

#endif /* _COXCOXCASCADENETWORK_H_ */
