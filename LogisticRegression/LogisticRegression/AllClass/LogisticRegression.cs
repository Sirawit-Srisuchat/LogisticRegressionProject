using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LogisticRegression.AllClass
{
    public class LogisticRegression
    {
        // Variables to store weights, learning rate, and the number of features
        private double[] weights;  // Weights for each feature (including bias)
        private double learningRate;  // Learning rate for gradient descent
        private int numFeatures;  // Number of features

        // Constructor to initialize the logistic regression model
        public LogisticRegression(int numFeatures, double learningRate = 0.1)
        {
            this.numFeatures = numFeatures;
            this.learningRate = learningRate;
            weights = new double[numFeatures + 1]; // +1 for the bias term
        }

        // Train the logistic regression model using gradient descent
        public void Train(double[][] X, int[] y, int numEpochs)
        {
            for (int epoch = 0; epoch < numEpochs; epoch++)
            {
                for (int i = 0; i < X.Length; i++)
                {
                    // Add bias term and make a prediction for the current example
                    double[] xWithBias = new double[X[i].Length + 1];
                    Array.Copy(X[i], 0, xWithBias, 1, X[i].Length);
                    xWithBias[0] = 1; // Bias term

                    double predicted = Predict(xWithBias);

                    // Compute the error and update the weights using gradient descent
                    double error = y[i] - predicted;

                    for (int j = 0; j < weights.Length; j++)
                    {
                        weights[j] += learningRate * error * xWithBias[j];
                    }
                }
            }
        }

        // Predict the output for given input features
        public double Predict(double[] features)
        {
            if (features.Length != numFeatures + 1)
                throw new ArgumentException("Invalid number of features.");

            double score = 0;
            for (int i = 0; i < features.Length; i++)
            {
                score += weights[i] * features[i];
            }

            return Sigmoid(score);
        }

        // Sigmoid activation function
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }
}
