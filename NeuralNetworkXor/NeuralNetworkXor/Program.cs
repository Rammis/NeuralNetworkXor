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


            Random random = new Random();
            float[,] inputs = new float[,] { { 60, 0, 0 }, { 0, 100, 0 }, { 0, 0, 40 } };
            float[,] outputs = new float[,] { { 1, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 } };
            float learningrate = 0.1f;
            int iterations = 1000;



            float[,] Wxh = new float[4, 3];
            float[,] Why = new float[4, 3];
            int t, i, h, o;


            for (i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                {
                    Wxh[i, j] = Convert.ToSingle(random.NextDouble() * (0.1 + 0.1) - 0.1);
                    Thread.Sleep(1);
                }
            for (i = 0; i < 4; i++)
                for (int j = 0; j < 3; j++)
                {
                    Why[i, j] = Convert.ToSingle(random.NextDouble() * (0.1 + 0.1) - 0.1);
                    Thread.Sleep(1);
                }

            float loss, sum; //for storing the loss


            float[] x = new float[4];
            float[] y = new float[3];
            float[] zhWeightedSums = new float[3];
            float[] hActivationValues = new float[4];
            float[] zyWeightedSums = new float[4];
            float[] probabilities = new float[3];
            float[] outputErrors = new float[3];
            float[,] deltaWxh = new float[4, 3];
            float[,] deltaWhy = new float[4, 3];





            for (t = 0; t < iterations; t++)
            {


                for (i = 1; i < 3; i++)
                    x[i] = inputs[t % 3, i - 1];
                x[0] = 1;

                for (i = 0; i < 3; i++)
                    y[i] = outputs[t % 3, i];



                for (i = 0; i < 3; i++)
                    zhWeightedSums[i] = 0;
                for (h = 0; h < 3; h++) for (i = 0; i < 3 + 1; i++) zhWeightedSums[h] += x[i] * Wxh[i, h];

                hActivationValues[0] = 1;
                for (h = 0; h < 3; h++) hActivationValues[h + 1] = Convert.ToSingle(Math.Tanh(zhWeightedSums[h]));
                for (i = 0; i < 4; i++)
                    zyWeightedSums[i] = 0;

                for (o = 0; o < 3; o++) for (h = 0; h < 3 + 1; h++) zyWeightedSums[o] += hActivationValues[h] * Why[h, o];

                for (sum = 0, o = 0; o < 3; o++) { probabilities[o] = Convert.ToSingle(Math.Exp(zyWeightedSums[o])); sum += probabilities[o]; }

                Console.WriteLine(probabilities[0] + " " + probabilities[1] + " " + probabilities[2]);
                for (o = 0; o < 3; o++) outputErrors[o] = probabilities[o] - y[o];

                for (loss = 0, o = 0; o < 3; o++) loss -= y[o] * Convert.ToSingle(Math.Log(probabilities[o]));
                Console.WriteLine(loss);



                for (h = 0; h < 3 + 1; h++) for (o = 0; o < 3; o++) deltaWhy[h, o] = hActivationValues[h] * outputErrors[o];



                for (i = 0; i < 4; i++)
                    hActivationValues[i] = 0;

                for (h = 1; h < 3 + 1; h++) for (o = 0; o < 3; o++) hActivationValues[h] += Why[h, o] * outputErrors[o];

                for (h = 0; h < 3; h++) zhWeightedSums[h] = hActivationValues[h + 1] * (1 - Convert.ToSingle(Math.Pow(Math.Tanh(zhWeightedSums[h]), 2)));


                for (i = 0; i < 3 + 1; i++) for (h = 0; h < 3; h++) deltaWxh[i, h] = x[i] * zhWeightedSums[h];


                for (h = 0; h < 3 + 1; h++) for (o = 0; o < 3; o++) Why[h, o] -= learningrate * deltaWhy[h, o];
                for (i = 0; i < 3 + 1; i++) for (h = 0; h < 3; h++) Wxh[i, h] -= learningrate * deltaWxh[i, h];


            }

            Console.ReadKey();

        }

    }
}
