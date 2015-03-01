#ifndef _ERRORFUNCTIONSGENERAL_HPP_
#define _ERRORFUNCTIONSGENERAL_HPP_

#include <vector>

void errorMSE(const std::vector<double> &Y,
              const unsigned int length,
              const unsigned int numOfOutput,
              const std::vector<double> &outputs,
              const unsigned int index,
              std::vector<double> &errors);

void derivativeMSE(const std::vector<double> &Y,
                   const unsigned int length,
                   const unsigned int numOfOutput,
                   const std::vector<double> &outputs,
                   const unsigned int index,
                   std::vector<double>::iterator result);

#endif /* _ERRORFUNCTIONSGENERAL_HPP_ */
