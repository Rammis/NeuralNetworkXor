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
            int[,] inputsData = new int[,] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
            int[,] expectedOutputData = new int[,] { { 0, 1 }, { 0, 1 }, { 0, 0 }, { 1, 0 } };
            Random random = new Random();
            double[,] firstLayerWeights = new double[4, 3]; //[X,0] for bias
            double[,] secondLayerWeights = new double[2, 5]; //[X,0] for bias
            int chosenInputs = random.Next(0, 3);
            int[] inputsFirstLayer = new int[3];
            inputsFirstLayer[0] = 1;
            int[] inputsSecondLayer = new int[5];
            double[] firstLayerSum = new double[4];
            double[] secondLayerSum = new double[2];
            double[] firstLayerSumActivate = new double[5];
            firstLayerSumActivate[0] = 1; //bias
            double[] secondLayerSumActivate = new double[2];
            double[] probabilities = new double[2];
            double[] outputErrors = new double[2];

            double[,] deltaSecondLayer = new double[2, 4]; //without bias
            double[,] deltaFirstLayer = new double[4, 2]; //without bias

            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                {
                    firstLayerWeights[i, j] = random.NextDouble();
                    Thread.Sleep(5);
                }
            for(int i=0;i<2;i++)
                for (int j = 0; j < 5; j++)
                {
                    secondLayerWeights[i, j] = random.NextDouble();
                    Thread.Sleep(5);
                }

            

            for (int i = 1; i < 3; i++)
                inputsFirstLayer[i] = inputsData[chosenInputs, i - 1];
            
            
            for (int i = 0; i < 4; i++)
            {
                firstLayerSum[i]=0;
                for (int j = 0; j < 3; j++)
                    firstLayerSum[i] += inputsFirstLayer[j] * firstLayerWeights[i, j];

                firstLayerSumActivate[i+1] = Math.Tanh(firstLayerSum[i]);
                    
            }

            for (int i = 0; i < 2; i++)
            {
                secondLayerSum[i]=0;
                for (int j = 0; j < 5; j++)
                    secondLayerSum[i] = firstLayerSumActivate[i] * secondLayerWeights[i, j];

                secondLayerSumActivate[i] = Math.Tanh(secondLayerSum[i]);
                    
            }
            
            double sum=0;
            for (int i = 0; i < 2; i++)
            {
                probabilities[i] = Math.Exp(secondLayerSumActivate[i]);
                sum += probabilities[i];
            }

            for (int i = 0; i < 2; i++)
                probabilities[i] /= sum;

            for (int i = 0; i < 2; i++)
                outputErrors[i] = expectedOutputData[chosenInputs, i] - probabilities[i];

                Console.WriteLine("Hello World");
            
           
        }
    }
}
