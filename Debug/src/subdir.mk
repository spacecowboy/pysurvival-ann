################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../src/ann.o 

CPP_SRCS += \
../src/FFNetwork.cpp \
../src/FFNeuron.cpp \
../src/GeneticSurvivalNetwork.cpp \
../src/RPropNetwork.cpp \
../src/activationfunctions.cpp \
../src/drand.cpp \
../src/simple_test.cpp 

OBJS += \
./src/FFNetwork.o \
./src/FFNeuron.o \
./src/GeneticSurvivalNetwork.o \
./src/RPropNetwork.o \
./src/activationfunctions.o \
./src/drand.o \
./src/simple_test.o 

CPP_DEPS += \
./src/FFNetwork.d \
./src/FFNeuron.d \
./src/GeneticSurvivalNetwork.d \
./src/RPropNetwork.d \
./src/activationfunctions.d \
./src/drand.d \
./src/simple_test.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -D__GXX_EXPERIMENTAL_CXX0X__ -I/usr/include/c++/4.6.3 -I/usr/include/boost/tr1 -O0 -g3 -Wall -c -fmessage-length=0 -std=c++0x -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


