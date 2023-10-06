using MathNet.Numerics.LinearAlgebra;  // Importing a library for linear algebra operations
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LogisticRegression.AllClass; // Import the corrected namespace

namespace LogisticRegression
{
    class Program
    {
        static void Main(string[] args)
        {
            // Sample data (features and labels)
            double[][] features = new double[][] {
                new double[] { 2.0, 1.0 },
                new double[] { 3.0, 1.5 },
                new double[] { 4.0, 2.0 },
                new double[] { 3.5, 0.5 },
                new double[] { 5.0, 2.0 },
                new double[] { 1.0, 1.0 }
            };

            int[] labels = new int[] { 0, 0, 0, 1, 1, 1 };

            // Create and train the logistic regression model
            LogisticRegression.AllClass.LogisticRegression logisticRegression =
                new LogisticRegression.AllClass.LogisticRegression(numFeatures: 2); // Initialize a logistic regression model

            // Train the logistic regression model using the provided sample data
            logisticRegression.Train(features, labels, numEpochs: 1000);

            // Test the model
            double[] testFeatures = new double[] { 2.5, 1.5 };
            double[] testFeaturesWithBias = new double[testFeatures.Length + 1];
            Array.Copy(testFeatures, 0, testFeaturesWithBias, 1, testFeatures.Length);
            testFeaturesWithBias[0] = 1; // Adding a bias term

            // Predict using the trained logistic regression model
            double prediction = logisticRegression.Predict(testFeaturesWithBias);

            // Display the prediction for the test features
            Console.WriteLine("Prediction for test features: " + prediction);
        }
    }
}
