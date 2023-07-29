#include <iostream>
#include <math.h>

class Neuron
{
public:
    int activate_relu();
    


    //Fast sigmoid function

private:
    //1.5
    double val;

    // 0-1
    double activatedVal;

    double derivedVal;
};