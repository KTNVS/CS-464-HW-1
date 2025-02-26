using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CS_464_HW_1
{
    public enum FeatureType : byte
    {
        Categorical,
        Continious
    }
    public class Feature(int key, int[] data, FeatureType type)
    {
        public readonly int Key = key;
        public readonly int[] Data = data;
        public readonly FeatureType Type = type;
    }

    public class NaiveBayes(int[] outputData) // allows output data to be categorical only
    {
        private readonly int[] OutputData = outputData;
        private readonly List<Feature> Features = [];

        private readonly int EntryCount = outputData.Length;
        private readonly double EntryCount_d = Convert.ToDouble(outputData.Length);


        private readonly Dictionary<int, double> PriorTable = [];
        private readonly Dictionary<int, Dictionary<int, Dictionary<int, double>>> FeatureCategoricalLikelihood = [];
        private readonly Dictionary<int, Dictionary<int, GaussianDistributionData>> FeatureContiniousLikelihood = [];

        public bool AddFeature(Feature Feature)
        {
            if(Feature.Data.Length != EntryCount)
            {
                Console.WriteLine($"Feature not added.\n" +
                    $"Feature entry count: {Feature.Data.Length} is not equal to output entry count: {EntryCount}");
                return false;
            }
            Features.Add(Feature);
            return true;
        }

        public readonly struct GaussianDistributionData(double mean, double standartDeviation)
        {
            public readonly double Mean = mean;
            public readonly double StandartDeviation = standartDeviation;
        }
        
        private void CalculateCategoricalLikelihood(Feature feature)
        {
            if(feature.Type != FeatureType.Categorical)
            {
                //
                return;
            }

            int[] UniqueOutputCategories = OutputData.Distinct().ToArray();

            Dictionary<int, Dictionary<int, double>> OutputLikelihoods = FeatureCategoricalLikelihood[feature.Key] = [];
            foreach (int outputType in UniqueOutputCategories)
            {
                Dictionary<int, double> categoryLikelihoods = OutputLikelihoods[outputType] = [];

                int outputCount = OutputData.Count(o => o == outputType);
                Dictionary<int, int> categoryCountsForThisOutput = feature.Data
                    .Where((f, i) => OutputData[i] == outputType)
                    .GroupBy(f => f)
                    .ToDictionary(f => f.Key, f => f.Count());

                foreach (var item in categoryCountsForThisOutput)
                    categoryLikelihoods[item.Key] = Convert.ToDouble(item.Value) / outputCount;
            }
        }
        private void CalculateContiniousLikelihood(Feature feature)
        {
            if (feature.Type != FeatureType.Continious)
            {
                return;
            }

            int[] UniqueOutputCategories = OutputData.Distinct().ToArray();

            Dictionary<int, GaussianDistributionData> OutputStatistics = FeatureContiniousLikelihood[feature.Key] = [];

            foreach (int outputType in UniqueOutputCategories)
            {
                double[] featureValues = feature.Data
                    .Where((f, i) => OutputData[i] == outputType)
                    .Select(Convert.ToDouble)
                    .ToArray();

                double mean = featureValues.Average();
                double variance = featureValues.Select(v => Math.Pow(v - mean, 2)).Average();
                double stdDev = Math.Sqrt(variance);

                OutputStatistics[outputType] = new GaussianDistributionData(mean, stdDev);
            }
        }
    }
}
