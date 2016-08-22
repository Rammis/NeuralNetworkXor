using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace NeuralNetworkXor
{
    class Program
    {
        static void Main(string[] args)
        {
            int[,] inputsData = new int[,] { { 1, 0,0,0 },{0,0,1,0},{0,0,0,1},{0,1,0,0}};
            int[,] expectedOutputData = new int[,] { { 0,0,0,1 }, {0,0,1,0}, { 0,1,0,0 }, {1,0,0,0} };
            Random random = new Random();
            double[,] firstLayerWeights = new double[4, 5]; //[X,0] for bias
            double[,] secondLayerWeights = new double[4, 5]; //[X,0] for bias
            int chosenInputs;
            int[] inputsFirstLayer = new int[5];
            inputsFirstLayer[0] = 1;
            int[] inputsSecondLayer = new int[5];
            double[] firstLayerSum = new double[4];
            double[] secondLayerSum = new double[4];
            double[] firstLayerSumActivate = new double[5];
            firstLayerSumActivate[0] = 1; //bias
            double loss = 0;
            
            double[] probabilities = new double[4];
            double[] outputErrors = new double[4];
            double[,] deltaSecondLayer = new double[4, 5]; //without bias
            double[,] deltaFirstLayer = new double[4, 5]; //without bias
            double sum = 0;
            double learningrate = 0.1F;

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 5; j++)
                {
                    firstLayerWeights[i, j] = random.NextDouble() * (0.1 + 0.1) - 0.1;
                    Thread.Sleep(5);
                }
            for(int i=0;i<2;i++)
                for (int j = 0; j < 5; j++)
                {
                    secondLayerWeights[i, j] = random.NextDouble() * (0.1 + 0.1) - 0.1;
                    Thread.Sleep(5);
                }



            for (int l = 0; l < 10000; l++)
            {
                chosenInputs = l % 4;

               
                for (int i = 1; i < 5; i++)
                    inputsFirstLayer[i] = inputsData[chosenInputs, i - 1];

                for (int i = 0; i < 4; i++)
                {
                    firstLayerSum[i] = 0;
                    for (int j = 0; j < 5; j++)
                        firstLayerSum[i] += inputsFirstLayer[j] * firstLayerWeights[i, j];

                    firstLayerSumActivate[i + 1] = Math.Tanh(firstLayerSum[i]);

                }
                firstLayerSumActivate[0] = 1;

                for (int i = 0; i < 4; i++)
                {
                    secondLayerSum[i] = 0;
                    for (int j = 0; j < 5; j++)
                        secondLayerSum[i] += firstLayerSumActivate[i] * secondLayerWeights[i, j];


                }

                sum = 0;
                for (int i = 0; i < 4; i++)
                {
                    probabilities[i] = Math.Exp(secondLayerSum[i]);
                    sum += probabilities[i];
                }

                for (int i = 0; i < 4; i++)
                    probabilities[i] /= sum;


                for (int i = 0; i < 4; i++)
                    outputErrors[i] = probabilities[i] - expectedOutputData[chosenInputs, i];
                loss = 0;
                for (int o = 0; o < 4; o++) loss -= expectedOutputData[chosenInputs,o] * Math.Log(probabilities[o]); //the loss

                Console.WriteLine(l + ": " + loss);

                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 5; j++)
                        deltaSecondLayer[i, j] = firstLayerSumActivate[j] * outputErrors[i];
                }

                for (int i = 0; i < 5; i++)
                    firstLayerSumActivate[i] = 0;

                for (int i = 0; i < 4; i++)
                {
                    for (int j = 1; j < 5; j++)
                        firstLayerSumActivate[j] += deltaSecondLayer[i, j] * outputErrors[i];
                }

                for (int i = 0; i < 4; i++)
                {
                    firstLayerSum[i] = firstLayerSumActivate[i + 1] * (1 - Math.Pow(Math.Tanh(firstLayerSum[i]), 2));
                }

                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 5; j++)
                        deltaFirstLayer[i, j] = inputsFirstLayer[j] * firstLayerSum[i];


               

                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 5; j++)
                        firstLayerWeights[i, j] -= learningrate * deltaFirstLayer[i, j];

                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 5; j++)
                        secondLayerWeights[i, j] -= learningrate * deltaSecondLayer[i, j];


                
            }

            Console.ReadKey();
           }
    }
}
