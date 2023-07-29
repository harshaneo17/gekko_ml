#include <iostream>
#include <vector>
//#include <../include/neuron.hpp>

struct Connection
{
    double weight;
    double deltaWeight;
};

class Neuron{};

typedef std::vector<Neuron> Layer;

class Neuron{
public:
    Neuron{};
private:
    double m_outputVal;
}

class Net{
public:
    Net(const std::vector<unsigned> &topology){
        unsigned numLayers = topology.size();
        for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
            m_layers.push_back(Layer());

            for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
                m_layers.back().push_back(Neuron());
            }
        }
    };
    void feedForward(const std::vector<double> &inputVals){};
    void backProp(const std::vector<double> &targetVals){};
    void getResults(std::vector<double> &targetVals) const{};

private:
    std::vector<Layer> m_layers;
};



int main(){
    std::vector<unsigned> topology;
    Net mynet(topology);

    std::vector<double> inputVals;
    mynet.feedForward(inputVals);

    std::vector<double> targetVals;
    mynet.backProp(targetVals);

    std::vector<double> resultVals;
    mynet.getResults(resultVals);
}