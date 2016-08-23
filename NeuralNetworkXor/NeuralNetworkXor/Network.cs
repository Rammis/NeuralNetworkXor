using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkXor
{
    class Network
    {
        int numberInputs;
        int numberOutputs;
        int numberHidden;
        int numberTraining;
        float learningrate = 0.005f;

        List<List<float>> inputs;
        List<List<float>> outputs;
        float[,] Wxh;
        float[,] Why;

        public Network(int numberInputs, int numberOutputs, int numberHidden, int numberTraining, List<List<float>> inputs, List<List<float>> outputs, float[,] Wxh, float[,] Why)
        {
            this.numberInputs = numberInputs;
            this.numberHidden = numberHidden;
            this.numberOutputs = numberOutputs;
            this.numberTraining = numberTraining;
            this.inputs = inputs;
            this.outputs = outputs;

            this.Wxh = new float[numberInputs + 1, numberHidden];
            this.Why = new float[numberHidden + 1, numberOutputs];
            this.Wxh = Wxh;
            this.Why = Why;
        }

        public void learning(int iterations, bool random)
        {
            float[] x = new float[numberInputs + 1];
            float[] y = new float[numberOutputs];
            float[] zhWeightedSums = new float[numberHidden];
            float[] hActivationValues = new float[numberHidden + 1];
            float[] zyWeightedSums = new float[numberOutputs];
            float[] probabilities = new float[numberOutputs];
            float[] outputErrors = new float[numberOutputs];
            float[,] deltaWxh = new float[numberInputs + 1, numberHidden];
            float[,] deltaWhy = new float[numberHidden + 1, numberOutputs];
            float loss, sum; //for storing the loss
            int t, i, h, o;
            Random randomTraining = new Random();
            int currentInputTraining;

            for (t = 0; t < iterations; t++)
            {

                if (random)
                    currentInputTraining = randomTraining.Next(0, numberTraining);
                else
                    currentInputTraining = t % numberTraining;

                for (i = 1; i < numberInputs; i++)
                    x[i] = Convert.ToSingle(inputs[currentInputTraining][i]);
                x[0] = 1;

                for (i = 0; i < numberOutputs; i++)
                    y[i] = outputs[currentInputTraining][i];



                for (i = 0; i < numberHidden; i++)
                    zhWeightedSums[i] = 0;
                for (h = 0; h < numberHidden; h++) for (i = 0; i < numberInputs + 1; i++) zhWeightedSums[h] += x[i] * Wxh[i, h];

                hActivationValues[0] = 1;
                for (h = 0; h < numberHidden; h++) hActivationValues[h + 1] = Convert.ToSingle(Math.Tanh(zhWeightedSums[h]));

                for (i = 0; i < numberOutputs; i++)
                    zyWeightedSums[i] = 0;

                for (o = 0; o < numberOutputs; o++) for (h = 0; h < numberHidden + 1; h++) zyWeightedSums[o] += hActivationValues[h] * Why[h, o];

                for (sum = 0, o = 0; o < numberOutputs; o++) { probabilities[o] = Convert.ToSingle(Math.Exp(zyWeightedSums[o])); sum += probabilities[o]; }

             

                for (o = 0; o < numberOutputs; o++) outputErrors[o] = probabilities[o] - y[o];

                for (loss = 0, o = 0; o < numberOutputs; o++) loss -= y[o] * Convert.ToSingle(Math.Log(probabilities[o]));
             
                for (h = 0; h < numberHidden + 1; h++) for (o = 0; o < numberOutputs; o++) deltaWhy[h, o] = hActivationValues[h] * outputErrors[o];


                for (i = 0; i < numberHidden + 1; i++)
                    hActivationValues[i] = 0;

                for (h = 1; h < numberHidden + 1; h++) for (o = 0; o < numberOutputs; o++) hActivationValues[h] += Why[h, o] * outputErrors[o];

                for (h = 0; h < numberHidden; h++) zhWeightedSums[h] = hActivationValues[h + 1] * (1 - Convert.ToSingle(Math.Pow(Math.Tanh(zhWeightedSums[h]), 2)));


                for (i = 0; i < numberInputs + 1; i++) for (h = 0; h < numberHidden; h++) deltaWxh[i, h] = x[i] * zhWeightedSums[h];


                for (h = 0; h < numberHidden + 1; h++) for (o = 0; o < numberOutputs; o++) Why
                [h, o] -= learningrate * deltaWhy[h, o];
                for (i = 0; i < numberInputs + 1; i++) for (h = 0; h < numberHidden; h++) Wxh[i, h] -= learningrate * deltaWxh[i, h];

            }
        }

        public List<List<float>> estimate(List<List<float>> inputs)
        {
            List<List<float>> allProbabilities = new List<List<float>>();

            float[] x = new float[numberInputs + 1];
            float[] y = new float[numberOutputs];
            float[] zhWeightedSums = new float[numberHidden];
            float[] hActivationValues = new float[numberHidden + 1];
            float[] zyWeightedSums = new float[numberOutputs];
            float[] probabilities = new float[numberOutputs];
            float sum; //for storing the loss
            int t, i, h, o;


            for (t = 0; t < inputs.Count; t++)
            {
                for (i = 1; i < numberInputs; i++)
                    x[i] = Convert.ToSingle(inputs[t][i]);
                x[0] = 1;

                for (i = 0; i < numberOutputs; i++)
                    y[i] = outputs[t][i];



                for (i = 0; i < numberHidden; i++)
                    zhWeightedSums[i] = 0;
                for (h = 0; h < numberHidden; h++) for (i = 0; i < numberInputs + 1; i++) zhWeightedSums[h] += x[i] * Wxh[i, h];

                hActivationValues[0] = 1;
                for (h = 0; h < numberHidden; h++) hActivationValues[h + 1] = Convert.ToSingle(Math.Tanh(zhWeightedSums[h]));

                for (i = 0; i < numberOutputs; i++)
                    zyWeightedSums[i] = 0;

                for (o = 0; o < numberOutputs; o++) for (h = 0; h < numberHidden + 1; h++) zyWeightedSums[o] += hActivationValues[h] * Why[h, o];

                List<float> currentProbabilities = new List<float>();
                for (sum = 0, o = 0; o < numberOutputs; o++) { probabilities[o] = Convert.ToSingle(Math.Exp(zyWeightedSums[o])); sum += probabilities[o];
                currentProbabilities.Add(probabilities[o]);
                }

                allProbabilities.Add(currentProbabilities.ToList());
            
            }
            return allProbabilities;
        }

        public void setLearningRate(float learningRate)
        {
            this.learningrate = learningRate;
        }
    }
}
