/*
 * A network which trains by use of the cascade correlation algorithm.
 * RProp is used to train  hidden layers, while output neuron is
 * replaced by a Cox proportional hazards model.
 */

#include "CoxCascadeNetwork.h"
#include "CascadeNetwork.h"
#include "RPropNetwork.h"
#include "FFNeuron.h"
#include "activationfunctions.h"
#include <vector>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include "c_index.h"
#include "rutil.h"
using namespace std;

CoxCascadeNetwork::CoxCascadeNetwork(unsigned int numOfInputs) :
  CascadeNetwork(numOfInputs, 1)
{

}

CoxCascadeNetwork::~CoxCascadeNetwork() {
  // Delete subclass specific allocations here
}

void CoxCascadeNetwork::initNodes() {
  hiddenRCascadeNeurons = new vector<RCascadeNeuron*>;

  this->hiddenNeurons = new Neuron*[0];
  unsigned int i;

  this->outputNeurons = new Neuron*[this->numOfOutput];
  for (i = 0; i < this->numOfOutput; i++) {
    this->outputNeurons[i] = new CoxNeuron(i);
  }

  this->bias = new RPropBias;
}

/*
 * Fit a cox model to the currently trained neurons and input data.
 */
void CoxCascadeNetwork::trainOutputs(double *X, double *Y, unsigned int rows) {
  unsigned int i;
  for (i = 0; i < this->numOfOutput; i++) {
    cout << "Calling fit\n";
    ((CoxNeuron*) this->outputNeurons[i])->fit(X, Y, rows);
  }
}

/*
 * Calculates the sum square error. Both the individual values,
 * and average value. Also stores the output values
 */
void CoxCascadeNetwork::calcErrors(double *X, double *Y, unsigned int rows,
                                double *patError, double *error,
                                double *outputs) {
  cout << "calcErrors\n";
  // Zero error array first
  memset(error, 0, numOfOutput * sizeof(double));

  // There is only one output neuron...
  cout << "calling output\n";
  ((CoxNeuron*) this->outputNeurons[0])->output(X, rows, outputs);
  cout << "calling getPatError\n";
  error[0] = getPatError(outputs, Y, rows, patError);
}

/**
 * Cox neuron definitions
 */

CoxNeuron::CoxNeuron(int id) :
  Neuron(id) {
  // Create R environment
  R = new RInside(0, NULL);
  R->parseEval("library(survival)");
}

CoxNeuron::~CoxNeuron() {
  // Destroy R environment
  R->parseEval("warnings()");
  delete R;
}

unsigned int CoxNeuron::getCoxNumOfCols() {
  // leading 2 is for target and event columns
  return 2 + this->inputConnections->size() + this->neuronConnections->size();
}

void CoxNeuron::convertToRColumns(double *X, double *Y, unsigned int rows,
                                  Rcpp::CharacterVector *colNames,
                                  vector<vector<double>*>
                                  *listOfColumns) {
  cout << "convertToRColumns\n";
  // Number of hidden neurons
  unsigned int nsize = this->neuronConnections->size();

  // Construct the array that holds the names of the columns in the data frame
  unsigned int i = 0;
  unsigned int j = 0;
  for (unsigned int col = 0; col < getCoxNumOfCols(); col++) {
    if (col == 0) {
      // Time column
      colNames->push_back("time");
      cout << "time, ";
    } else if (col == 1) {
      // Event column
      colNames->push_back("event");
      cout << "event, ";
    } else if (col < 2 + this->inputConnections->size()) {
      // Input columns
      stringstream sstm;
      sstm << "X" << i;
      i++;
      // Add name
      colNames->push_back(sstm.str());
      cout << sstm.str() << ", ";
    } else {
      // Neuron connections
      stringstream sstm;
      sstm << "N" << j;
      j++;
      // Add name
     colNames->push_back(sstm.str());
     cout << sstm.str() << ", ";
    }
  }
  cout << "\n";

  // Construct the actual matrix of the data frame
  for (unsigned int row = 0; row < rows; row++) {
    cout << "\n";
    int col = 0;
    // time column
    if (Y == NULL)
      listOfColumns->at(col)->push_back(0);
    else
      listOfColumns->at(col)->push_back(Y[2*row]);
    cout << listOfColumns->at(col)->back() << ", ";
    col++;
    // event column
    if (Y == NULL)
      listOfColumns->at(col)->push_back(0);
    else
      listOfColumns->at(col)->push_back(Y[2*row + 1]);
    cout << listOfColumns->at(col)->back() << ", ";
    col++;
    // input columns
    for (i = 0; i < this->inputConnections->size(); i++) {
      listOfColumns->at(col)->push_back(X[i + row*this->inputConnections->size()]);
      cout << listOfColumns->at(col)->back() << ", ";
      col++;
    }
    // neuron columns
    // Using fact that neurons are placed in a chain and thus
    // can be evaluated as we loop over them.
    for (j = 0; j < nsize; j++) {
      listOfColumns->at(col)->push_back(neuronConnections->at(j)
                                        .first->output(X + row*this->inputConnections->size()));
      cout << listOfColumns->at(col)->back() << ", ";
      col++;
    }
  }
  cout << "\n";
}

string getCoxCmd(Rcpp::CharacterVector *colNames) {
  stringstream sstm;

  // coxfit <- coxph(Surv(time, event) ~ X0
  sstm << "coxfit <- coxph(Surv(" << (*colNames)[0] << ", "
       << (*colNames)[1] << ") ~ " << (*colNames)[2];
  for (int c = 0; c < colNames->size(); c++) {
    // +X1,X2,X3,...N0,N1,N2...
    sstm << "+" << (*colNames)[c];
  }

  // , data, model=TRUE)
  sstm << ", data, model=TRUE)";

  // Also add summary command
  sstm << "; s <- summary(coxfit)";

  /*
    coxfit <- coxph(Surv(time, event) ~ X0+X1+X2+N0+N1+N2, data, model=TRUE)
    s <- summary(coxfit)
  */
  cout << "coxCmd: " << sstm.str() << "\n";
  return sstm.str();
}

void CoxNeuron::fit(double *X, double *Y, unsigned int rows) {
  cout << "fit\n";
  // Create a cox model for the data and connected neurons
  // Cox model is stored as variable in R-environment
  cout << "doing names\n";
  // First convert the data to R-compatible data frame
  vector<vector<double>*> colList;
  for (unsigned int i = 0; i < getCoxNumOfCols(); i++) {
    colList.push_back(new vector<double>);
  }
  Rcpp::CharacterVector colNames;
  cout << "doing convertToRMatrix\n";
  convertToRColumns(X, Y, rows, &colNames, &colList);
  Rcpp::DataFrame dd = getDataFrameWithNames(&colNames, &colList);
  cout << "setting data in R\n";
  (*R)["data"] = dd;
  cout << "Get and execute cox cmd\n";
  // Create and fit the cox model
  R->parseEval(getCoxCmd(&colNames));

  cout << "Destroy data\n";
  // destroy data
  for (unsigned int i = 0; i < getCoxNumOfCols(); i++) {
    delete colList.back();
    colList.pop_back();
  }
  cout << "Data destroyed\n";
  R->parseEval("warnings()");
}

double CoxNeuron::output(double *inputs) {
  return output(inputs, 1, NULL);
}

double CoxNeuron::output(double *inputs, unsigned int rows, double *outputs) {
  cout << "output\n";
  // Ask the cox model to predict from the input data and connected neurons

  // First convert the data to R-compatible data frame
  vector<vector<double>*> colList;
  for (unsigned int i = 0; i < getCoxNumOfCols(); i++) {
    colList.push_back(new vector<double>);
  }
  Rcpp::CharacterVector colNames;
  cout << "convertToMatrix\n";
  convertToRColumns(inputs, NULL, rows, &colNames, &colList);
  Rcpp::DataFrame dd = getDataFrameWithNames(&colNames, &colList);
  cout << "set data\n";
  (*R)["pred_data"] = dd;
  cout << "call predict\n";
  // Predict outcomes
  Rcpp::NumericVector v = R->parseEval("predict(coxfit, newdata=pred_data, type=\"lp\")");
  cout << "set result\n";
  if (outputs != NULL) {
    for (unsigned int i = 0; i < rows; i++) {
      outputs[i] = v[i];
    }
  }
  cout << "destroy data\n";
  // destroy data
  for (unsigned int i = 0; i < getCoxNumOfCols(); i++) {
    delete colList.back();
    colList.pop_back();
  }

  // Return the prediction
  return v[0];
}

double CoxNeuron::getConcordance() {
  // Get the c-index calculated when trained from R
  Rcpp::NumericVector v = R->parseEval("s$concordance");
  // C-index in first element
  return v[0];
}
