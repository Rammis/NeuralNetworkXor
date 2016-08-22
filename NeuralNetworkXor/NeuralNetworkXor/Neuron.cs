using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkXor
{
    class Neuron
    {
        double[] inputs { get; set; }
        double[] weights { get; set;}
        double sumWeightedOutputs;
        double activateOutput;

        public Neuron(double[] inputs, double[] weights)
        {
            this.inputs = inputs;
            this.weights = weights;
        }

        double getSumWeightedOutputs()
        {
            sumWeightedOutputs = 0;

            for (int i = 0; i < inputs.Length; i++)
                sumWeightedOutputs += inputs[i] * weights[i];

            return sumWeightedOutputs;
        }

        double getActivateOutput()
        {
            return Math.Tanh(getSumWeightedOutputs());
        }




    }
}
