#include <iostream>
#include <vector>
#include <C:\Users\ASUS\Documents\cpp_projects\average_engine_ml\include\neuron.hpp>




class Net{
public:
    Net(const std::vector<unsigned> &topology){
        unsigned numLayers = topology.size();
        for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
            /* This takes number of layers as the length of the topology vector 
            and creates a fills the vector m_layers with layers in a for loop */
            m_layers.push_back(Layer());
            unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

            for (unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
                m_layers.back().push_back(Neuron());
            }
        }
    };
    void feedForward(const std::vector<double> &inputVals){};
    void backProp(const std::vector<double> &targetVals){};
    void getResults(std::vector<double> &targetVals) const{};

private:
    std::vector<Layer> m_layers; //this is a vector within a vector m_layers[layerNum][neuronNum]
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