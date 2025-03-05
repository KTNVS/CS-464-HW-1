namespace CS_464_HW_1
{
    public static class FeatureSelection
    {
        public static List<(int FeatureIndex, double MutualInformationValue)> GetFeatureIndexesOrderedByMutualInformation(DataXY data)
        {
            int featureCount = data.FeatureData.RowCount;

            List<(int FeatureIndex, double MutualInformationValue)> featureMutualInformationList = [];

            for (int i = 0; i < featureCount; i++)
            {
                double mutualInformationValue = GetMutualInformation(data.FeatureData.GetRow(i), data.OutputData);
                featureMutualInformationList.Add((i, mutualInformationValue));
            }

            return [.. featureMutualInformationList.OrderByDescending(f => f.MutualInformationValue)];
        }

        private static double GetMutualInformation(int[] featureData, int[] outputData)
        {
            double mutualInformation = 0d;
            int entryCount = featureData.Length;
            
            var featureFreq = new Dictionary<int, int>();
            var outputFreq = new Dictionary<int, int>();
            var jointFreq = new Dictionary<(int, int), int>();

            for (int i = 0; i < entryCount; i++)
            {
                int f = featureData[i];
                int o = outputData[i];

                featureFreq[f] = featureFreq.GetValueOrDefault(f, 0) + 1;
                outputFreq[o] = outputFreq.GetValueOrDefault(o, 0) + 1;
                jointFreq[(f, o)] = jointFreq.GetValueOrDefault((f, o), 0) + 1;
            }

            foreach (var ((f, o), jointCount) in jointFreq)
            {
                double pFeatureOutput = jointCount / (double)entryCount;
                double pFeature = featureFreq[f] / (double)entryCount;
                double pOutput = outputFreq[o] / (double)entryCount;

                mutualInformation += pFeatureOutput * Math.Log2(pFeatureOutput / (pFeature * pOutput));
            }

            return mutualInformation;
        }
    }
}
