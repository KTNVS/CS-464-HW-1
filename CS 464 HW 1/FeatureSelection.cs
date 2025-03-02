using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CS_464_HW_1
{
    public static class FeatureSelection
    {
        public static void PrintMutualInformation(DataXY data, List<string> featureNames)
        {
            int featureCount = data.FeatureData.RowCount;
            List<(string FeatureName, double MutualInformationValue)> featureMiList = [];

            for (int i = 0; i < featureCount; i++)
            {
                double mutualInformationValue = GetMutualInformation(data.FeatureData.GetRow(i), data.OutputData);
                featureMiList.Add((featureNames[i], mutualInformationValue));
            }

            var sortedFeatures = featureMiList.OrderByDescending(f => f.MutualInformationValue).ToList();

            Console.WriteLine("Mutual Information Values:");
            foreach (var (FeatureName, MutualInformationValue) in sortedFeatures)
                Console.WriteLine($"Feature type => {FeatureName}: {MutualInformationValue:N10}");
        }

        private static double GetMutualInformation(int[] featureData, int[] outputData)
        {
            double mutualInformation = 0d;

            int entryCount = featureData.Length;

            foreach (int f in featureData.Distinct())
            {
                foreach (int o in outputData.Distinct())
                {
                    // Calculate P(feature, output)
                    double pFeatureOutput = featureData
                        .Zip(outputData, (f, o) => (f, o))
                        .Count(pair => pair.f == f && pair.o == o) / (double)entryCount;

                    // Calculate P(feature) and P(output)
                    double pFeature = featureData.Count(fv => fv == f) / (double)entryCount;
                    double pOutput = outputData.Count(ov => ov == o) / (double)entryCount;

                    // Update mutual information if pFeatureOutput > 0
                    if (pFeatureOutput > 0)
                        mutualInformation += pFeatureOutput * Math.Log2(pFeatureOutput / (pFeature * pOutput));
                }
            }
            return mutualInformation;
        }
    }
}
