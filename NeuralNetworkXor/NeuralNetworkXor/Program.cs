using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using NeuralNetworkXor;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            int numberInputs = 2;
            int numberHidden = 10;
            int numberOutputs = 2;
            int numberTraining = 4;
            int i;
            Random random = new Random();
            float[,] Wxh = new float[numberInputs + 1, numberHidden];
            float[,] Why = new float[numberHidden + 1, numberOutputs];

            List<List<float>> inputs = new List<List<float>>();
            List<List<float>> outputs = new List<List<float>>();

            List<float> temp = new List<float>();

            temp.Add(2);
            temp.Add(4);

            inputs.Add(temp.ToList());
            temp.Clear();

            
            temp.Add(5);
            temp.Add(6);

            inputs.Add(temp.ToList());
            temp.Clear();

           
            temp.Add(7);
            temp.Add(3);

            inputs.Add(temp.ToList());
            temp.Clear();

            
            temp.Add(-2);
            temp.Add(-4);

            inputs.Add(temp.ToList());
            temp.Clear();

            temp.Add(1);
            temp.Add(0);
            outputs.Add(temp.ToList());
            temp.Clear();

            temp.Add(1);
            temp.Add(0);
            outputs.Add(temp.ToList());
            temp.Clear();

            temp.Add(1);
            temp.Add(0);
            outputs.Add(temp.ToList());
            temp.Clear();

            temp.Add(0);
            temp.Add(1);
            outputs.Add(temp.ToList());
            temp.Clear();


            for (i = 0; i < numberInputs + 1; i++)
            {
                List<float> tmp = new List<float>();
                for (int j = 0; j < numberHidden; j++)
                {
                    Wxh[i, j] = Convert.ToSingle(random.NextDouble() * (0.1 + 0.1) - 0.1);
                    Thread.Sleep(1);
                }

            }
            for (i = 0; i < numberHidden + 1; i++)
            {
                List<float> tmp = new List<float>();
                for (int j = 0; j < numberOutputs; j++)
                {
                    Why[i, j] = Convert.ToSingle(random.NextDouble() * (0.1 + 0.1) - 0.1);
                    Thread.Sleep(1);
                }

            }

            Network myNetwork = new Network(numberInputs, numberOutputs, numberHidden, numberTraining, inputs, outputs, Wxh, Why);
            myNetwork.learning(1000000, false);

            List<List<float>> probabilities = myNetwork.estimate(inputs);
        
        }

    }
}
