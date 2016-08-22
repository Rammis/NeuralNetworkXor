using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            int numberInputs = 5;
            int numberHidden = 5;
            int numberOutputs = 5;
            int numberTraining = 10;

            Random random = new Random();
            float[,] inputs = new float[,] {{10,20,3,4}, {18,77,8,6}, {5,4,6,6}, {-1,-8,-3,-5}};
            float[,] outputs = new float[,] {{1,0}, {1,0}, {1,0}, {0,1}};
            float learningrate = 0.05f;
            int iterations = 10000;



            float[,] Wxh = new float[numberInputs+1, numberHidden];
            float[,] Why = new float[numberHidden + 1, numberOutputs];
            int t, i, h, o;


            for (i = 0; i < numberInputs + 1; i++)
                for (int j = 0; j < numberHidden; j++)
                {
                    Wxh[i, j] = Convert.ToSingle(random.NextDouble() * (0.1 + 0.1) - 0.1);
                    Thread.Sleep(1);
                }
            for (i = 0; i < numberHidden + 1; i++)
                for (int j = 0; j < numberOutputs; j++)
                {
                    Why[i, j] = Convert.ToSingle(random.NextDouble() * (0.1 + 0.1) - 0.1);
                    Thread.Sleep(1);
                }

            float loss, sum; //for storing the loss


            float[] x = new float[numberInputs+1];
            float[] y = new float[numberOutputs];
            float[] zhWeightedSums = new float[numberHidden];
            float[] hActivationValues = new float[numberHidden+1];
            float[] zyWeightedSums = new float[numberOutputs];
            float[] probabilities = new float[numberOutputs];
            float[] outputErrors = new float[numberOutputs];
            float[,] deltaWxh = new float[numberInputs+1, numberHidden];
            float[,] deltaWhy = new float[numberHidden+1, numberOutputs];





            for (t = 0; t < iterations; t++)
            {

                int counter = 0;
                List<String> dates = new List<String>();
                List<double> inputsList = new List<double>();
                String outputList = "";
                string line;
                int numberFile = t % numberTraining;
                string fileName = numberFile.ToString() + ".txt";
                
                // Read the file and display it line by line.
                System.IO.StreamReader file =
                    new System.IO.StreamReader(@fileName);
                while ((line = file.ReadLine()) != null)
                {
                    if (counter % 2 == 0 && counter != 10)
                    {
                        dates.Add(line);
                    }
                    else if (counter % 2 == 1)
                        inputsList.Add(double.Parse(line));
                    else
                        outputList = line;
                    
                    counter++;
                }

                file.Close();

                for (i = 1; i < numberInputs; i++)
                    x[i] = Convert.ToSingle(inputsList[i - 1]);
                x[0] = 1;

                for (i = 0; i < numberOutputs; i++)
                    y[i] = outputList[i] - 48;



                for (i = 0; i < numberHidden; i++)
                    zhWeightedSums[i] = 0;
                for (h = 0; h < numberHidden; h++) for (i = 0; i < numberHidden + 1; i++) zhWeightedSums[h] += x[i] * Wxh[i, h];

                hActivationValues[0] = 1;
                for (h = 0; h < numberHidden; h++) hActivationValues[h + 1] = Convert.ToSingle(Math.Tanh(zhWeightedSums[h]));
                
                for (i = 0; i < numberOutputs; i++)
                    zyWeightedSums[i] = 0;

                for (o = 0; o < numberOutputs; o++) for (h = 0; h < numberHidden + 1; h++) zyWeightedSums[o] += hActivationValues[h] * Why[h, o];

                for (sum = 0, o = 0; o < numberOutputs; o++) { probabilities[o] = Convert.ToSingle(Math.Exp(zyWeightedSums[o])); sum += probabilities[o]; }

                Console.WriteLine(numberFile + ": ");
                for (o = 0; o < numberOutputs; o++)
                    Console.Write(probabilities[o] + " ");
                Console.WriteLine();
                for (o = 0; o < numberOutputs; o++)
                    Console.Write(y[o] + " ");

                    for (o = 0; o < numberOutputs; o++) outputErrors[o] = probabilities[o] - y[o];

                for (loss = 0, o = 0; o < numberOutputs; o++) loss -= y[o] * Convert.ToSingle(Math.Log(probabilities[o]));
                Console.WriteLine();
                Console.WriteLine("Loss: " + loss);



                for (h = 0; h < numberHidden + 1; h++) for (o = 0; o < numberOutputs; o++) deltaWhy[h, o] = hActivationValues[h] * outputErrors[o];


                for (i = 0; i < numberInputs + 1; i++)
                    hActivationValues[i] = 0;

                for (h = 1; h < numberHidden + 1; h++) for (o = 0; o < numberOutputs; o++) hActivationValues[h] += Why[h, o] * outputErrors[o];

                for (h = 0; h < numberHidden; h++) zhWeightedSums[h] = hActivationValues[h + 1] * (1 - Convert.ToSingle(Math.Pow(Math.Tanh(zhWeightedSums[h]), 2)));


                for (i = 0; i < numberInputs + 1; i++) for (h = 0; h < numberHidden; h++) deltaWxh[i, h] = x[i] * zhWeightedSums[h];


                for (h = 0; h < numberHidden + 1; h++) for (o = 0; o < numberOutputs; o++) Why
                [h, o] -= learningrate * deltaWhy[h, o];
                for (i = 0; i < numberInputs + 1; i++) for (h = 0; h < numberHidden; h++) Wxh[i, h] -= learningrate * deltaWxh[i, h];


            }

            Console.ReadKey();

        }

    }
}
