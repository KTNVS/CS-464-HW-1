namespace CS_464_HW_1
{
    public enum EstimationType : byte
    {
        Categorical,
        Gaussian
    }
    public class Feature(int featureIndex, int[] data, EstimationType estimation)
    {
        public readonly int Index = featureIndex; // ID, dictionary key
        public readonly int[] Data = data;
        public readonly EstimationType EstimationType = estimation;
    }
    public class Output(int[] outputData)
    {
        public readonly int[] Data = outputData;
        public readonly int[] UniqueCategories = outputData.Distinct().ToArray();
        public readonly Dictionary<int, double> LikelihoodSet =
            outputData.GroupBy(o => o).ToDictionary(o => o.Key, o => Convert.ToDouble(o.Count()) / Convert.ToDouble(outputData.Length));
    }

    public class NaiveBayes(int[] outputData)
    {
        public static readonly double ALPHA_SMOOTHING_COFACTOR = Math.Log(1e-6);

        private readonly int EntryCount = outputData.Length;
        private readonly Output Output = new(outputData);

        private readonly Dictionary<int, ILikelihoodEstimator> FeatureEstimators = [];
        public void PrintFeatureEstimationInfo(int i) => FeatureEstimators[i].PrintInfo();
        public void ResetEstimationData() => FeatureEstimators.Clear();

        public void LearnFeature(Feature Feature) // since bayes considers them independent, no problem learning seperately
        {
            if(Feature.Data.Length != EntryCount)
            {
                Console.WriteLine($"[WARNING] Feature not added.\n" +
                    $"Feature entry count: {Feature.Data.Length} is not equal to output entry count: {EntryCount} in (AddFeature)");
                return;
            }
            ILikelihoodEstimator Estimator = FeatureEstimators[Feature.Index] 
                = Feature.EstimationType == EstimationType.Categorical ? new CategoricalEstimator() : new GaussianEstimator();
            Estimator.EvaluateData(Feature.Data, Output);
        }
        public int? PredictFeature(int[] featureValues)
        {
            int? bestOutput = null;
            double bestLogProbability = double.NegativeInfinity;

            foreach (int outputType in Output.UniqueCategories)
            {
                double logProbability = Math.Log(Output.LikelihoodSet[outputType]);

                for (int i = 0; i < featureValues.Length; i++)
                {
                    if (FeatureEstimators.TryGetValue(i, out ILikelihoodEstimator? estimator) && estimator != null)
                        logProbability += estimator.GetProbability(featureValues[i], outputType);
                }
                if (logProbability > bestLogProbability)
                {
                    bestLogProbability = logProbability;
                    bestOutput = outputType;
                }
            }
            return bestOutput;
        }
    }

    #region FeatureLikelihoodEstimators
    public interface ILikelihoodEstimator
    {
        public void EvaluateData(int[] featureData, Output output);
        public double GetProbability(int featureValue, int outputValue);
        public void PrintInfo();
    }
    public class CategoricalEstimator : ILikelihoodEstimator
    {
        private readonly Dictionary<int, Dictionary<int, double>> CategoricalLikelihoodMatrix = []; // output, category
        public void EvaluateData(int[] featureData, Output output)
        {
            foreach (int outputType in output.UniqueCategories)
            {
                Dictionary<int, double> categoryLikelihoods = CategoricalLikelihoodMatrix[outputType] = [];

                int outputCount = output.Data.Count(o => o == outputType);
                Dictionary<int, int> categoryCountsForThisOutput = featureData
                    .Where((f, i) => output.Data[i] == outputType)
                    .GroupBy(f => f)
                    .ToDictionary(f => f.Key, f => f.Count());

                foreach (var item in categoryCountsForThisOutput)
                    categoryLikelihoods[item.Key] = Convert.ToDouble(item.Value) / outputCount;
            }
        }
        public double GetProbability(int featureValue, int outputValue)
        {
            if (CategoricalLikelihoodMatrix.TryGetValue(outputValue, out var categoryLikelihoods) && categoryLikelihoods.TryGetValue(featureValue, out double probability))
                return Math.Log(probability);

            return NaiveBayes.ALPHA_SMOOTHING_COFACTOR;
        }

        public void PrintInfo()
        {
            foreach (var output in CategoricalLikelihoodMatrix)
            {
                Console.WriteLine($"Output category: {output.Key}");
                foreach (var category in output.Value.OrderByDescending(f => f.Value))
                    Console.WriteLine($"\tCategory [{category.Key}] probability: {category.Value:N6}");
            }
        }
    }
    public class GaussianEstimator : ILikelihoodEstimator
    {
        private readonly Dictionary<int, GaussianDistributionData> ContinuousLikelihoodMatrix = [];

        public void EvaluateData(int[] featureData, Output output)
        {
            foreach (int outputType in output.UniqueCategories)
            {
                double[] featureValues = featureData
                    .Where((f, i) => output.Data[i] == outputType)
                    .Select(Convert.ToDouble)
                    .ToArray();

                if (featureValues.Length == 0)
                {
                    ContinuousLikelihoodMatrix[outputType] = GaussianDistributionData.Default;
                    continue;
                }

                double mean = featureValues.Average();
                double variance = featureValues.Select(v => Math.Pow((double)v - mean, 2d)).Sum() / Math.Max(featureValues.Length - 1, 1d);

                ContinuousLikelihoodMatrix[outputType] = new GaussianDistributionData(mean, variance);
            }
        }

        public double GetProbability(int featureValue, int outputValue)
        {
            if (ContinuousLikelihoodMatrix.TryGetValue(outputValue, out var gaussian))
            {
                double exponent = -Math.Pow(Convert.ToDouble(featureValue) - gaussian.Mean, 2) / (2 * gaussian.Variance);
                double probability = Math.Exp(exponent) / (Math.Sqrt(2 * Math.PI * gaussian.Variance));
                return Math.Log(probability);
            }
            return NaiveBayes.ALPHA_SMOOTHING_COFACTOR;
        }

        private readonly struct GaussianDistributionData(double mean, double variance)
        {
            public readonly double Mean = mean;
            public readonly double Variance = variance;

            public static GaussianDistributionData Default = new(0, 1);
        }

        public void PrintInfo()
        {
            Console.WriteLine("Continious info");
        }
    }


    #endregion
}
