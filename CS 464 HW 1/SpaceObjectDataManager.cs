namespace CS_464_HW_1
{
    public class DataXY
    {
        public int EntryCount;
        public DataMatrix<int> FeatureData = DataMatrix<int>.Empty;
        public int[] OutputData = [];
    }
    public class SpaceObjectDataManager
    {
        const int INT_LABEL_LENGTH = 2;
        const int UNDOCUMENTED_VALUE_NUMBER = -9999; // will be replaced with mean value if its continuous

        private readonly NaiveBayes NaiveBayes;
        private bool FeatuesEvaluated = false;
        private readonly bool SelectGrouped;

        public const int FEATURE_COUNT_DEFAULT = 9, FEATURE_COUNT_GROUPED = 6;
        public readonly int FeatureCount;
        public static readonly EstimationType[] FeatureTypes =
        [
            EstimationType.Categorical,     // redshift
            EstimationType.Categorical,     // alpha
            EstimationType.Categorical,     // delta
            EstimationType.Gaussian,        // green_filter
            EstimationType.Gaussian,        // near_infrared_filter
            EstimationType.Categorical,     // cosmic_ray_activity
            EstimationType.Gaussian,        // red_filter
            EstimationType.Gaussian,        // ultraviolet_filter
            EstimationType.Gaussian,        // infrared_filter
        ];

        private readonly DataXY TrainData = new();
        private readonly DataXY TestData = new();

        public SpaceObjectDataManager(string trainFeaturesPath, string trainOutputPath, string testFeaturesPath, string testOutputPath, bool selectGrouped = false) 
        {
            FeatureCount = selectGrouped ? FEATURE_COUNT_GROUPED : FEATURE_COUNT_DEFAULT;
            SelectGrouped = selectGrouped;
            
            try
            {
                TrainData = ExtractDataFromCSV(trainFeaturesPath, trainOutputPath);
                TestData = ExtractDataFromCSV(testFeaturesPath, testOutputPath);
            }
            catch { throw; }

            RemoveUndocumentedValues(TrainData.FeatureData);
            RemoveUndocumentedValues(TestData.FeatureData);

            // array of object features => array of feature arrays; not done for test data as single objects outputs are predicted
            TrainData.FeatureData = TrainData.FeatureData.Transpose();
            NaiveBayes = new(TrainData.OutputData);
        }
        public void ResetEvaluationData()
        {
            NaiveBayes.ResetEstimationData();
            FeatuesEvaluated = false;
        }
        public void Evaluate(bool forceCategorical = true, int selectKMutuals = 0) // 0, select all, 1-9 select respectively, 
        {
            var features = (selectKMutuals >= 1 && selectKMutuals <= FeatureCount - 1)
                ? FeatureSelection.GetFeatureIndexesOrderedByMutualInformation(TrainData).Take(selectKMutuals).Select(f => f.FeatureIndex)
                : Enumerable.Range(0, TrainData.FeatureData.RowCount);

            Console.Write("Evaluation started...");
            foreach (var index in features)
                NaiveBayes.LearnFeature(new Feature(index, TrainData.FeatureData.GetRow(index), forceCategorical ? EstimationType.Categorical : FeatureTypes[index]));

            Console.WriteLine("Evaluation completed.");
            FeatuesEvaluated = true;
        }
        public void Predict()
        {
            Console.Write("Prediction started...");
            int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
            for (int i = 0; i < TestData.EntryCount; i++)
            {
                int? predictedOutput = NaiveBayes.PredictFeature(TestData.FeatureData.GetRow(i));
                if(predictedOutput == null)
                {
                    Console.WriteLine($"\n[Warning] Output couldn't be predicted for object index {i}. Skipping.");
                    return;
                }

                if(TestData.OutputData[i] == predictedOutput)
                {
                    if (predictedOutput == 1)
                        truePositive++;
                    else
                        trueNegative++;
                }
                else
                {
                    if (predictedOutput == 1)
                        falsePositive++;
                    else
                        falseNegative++;
                }
            }
            Console.WriteLine("Prediction completed.");

            double total = Convert.ToDouble(truePositive + trueNegative + falsePositive + falseNegative);
            Console.WriteLine($"\nTotal accuracy: {Convert.ToDouble(truePositive + trueNegative) / total * 100d}%");
            Console.WriteLine($"\nTrue Positive: {Convert.ToDouble(truePositive) / total * 100d}%");
            Console.WriteLine($"True Negative: {Convert.ToDouble(trueNegative) / total * 100d}%");
            Console.WriteLine($"False Positive: {Convert.ToDouble(falsePositive) / total * 100d}%");
            Console.WriteLine($"False Negative: {Convert.ToDouble(falseNegative) / total * 100d}%\n\n");
        }
        public void PrintAllFeaturesCategoryCount()
        {
            Console.WriteLine($"Output category count: {TrainData.OutputData.Distinct().Count()}");
            for (int i = 0; i < TrainData.FeatureData.RowCount; i++)
                Console.WriteLine($"Feature [{FeatureNames[i]}] category count: {TrainData.FeatureData.GetRow(i).Distinct().Count()}");
        }
        public void PrintFeatureProbabilities()
        {
            if (!FeatuesEvaluated)
            {
                Console.WriteLine("Can't show feature probabilities, features are not evaluated.");
                return;
            }
            for (int i = 0; i < TrainData.FeatureData.RowCount; i++)
            {
                Console.WriteLine($"\n\nFeature: {FeatureNames[i]}\n");
                NaiveBayes.PrintFeatureEstimationInfo(i);
            }
        }
        public void PrintMutualInformation() 
        {
            var sortedFeatures = FeatureSelection.GetFeatureIndexesOrderedByMutualInformation(TrainData);
            Console.WriteLine("Mutual Information Values:");
            foreach (var (FeatureIndex, MutualInformationValue) in sortedFeatures)
                Console.WriteLine($"Feature type => {FeatureNames[FeatureIndex]}: {MutualInformationValue:N10}");
            Console.WriteLine();
        }
            

        private void RemoveUndocumentedValues(DataMatrix<int> data)
        {
            for (int featureIndex = 0; featureIndex < data.ColCount; featureIndex++)
            {
                if (FeatureTypes[featureIndex] == EstimationType.Gaussian)
                {
                    int average = Convert.ToInt32(data.GetCol(featureIndex).Average());
                    for (int row = 0; row < data.RowCount; row++)
                        if (data[row, featureIndex] == UNDOCUMENTED_VALUE_NUMBER)
                            data[row, featureIndex] = average;
                }
            }
        }
        private DataXY ExtractDataFromCSV(string FeaturesPath, string outputPath)
        {
            DataXY csvData = new();
            DataMatrix<string> FeatureMatrix;
            try
            {
                csvData.OutputData = CSVReader.ReadRows(outputPath, false).Select(n => n.Equals("True") ? 1 : 0).ToArray();
                FeatureMatrix = CSVReader.ReadCSV(FeaturesPath, true);
            }
            catch { throw; }

            if (csvData.OutputData.Length != FeatureMatrix.RowCount)
                throw new Exception($"[ERROR] X data length: {FeatureMatrix.RowCount} and Y data length: {outputPath.Length} are not the same.");
            if (FEATURE_COUNT_DEFAULT != FeatureMatrix.ColCount)
                throw new Exception($"[ERROR] X feature count: {FeatureMatrix.ColCount} is not equal to the number of expected features of: {FEATURE_COUNT_DEFAULT}");

            csvData.EntryCount = csvData.OutputData.Length;

            int[][] objectFeatures = new int[FeatureMatrix.RowCount][];
            for (int row = 0; row < csvData.EntryCount; row++)
            {
                int[] rowData = ConvertRowData(FeatureMatrix.GetRow(row));
                if (rowData.Length == 0)
                {
                    Console.WriteLine($"Skipping row {row}.");
                    continue;
                }
                objectFeatures[row] = rowData;
            }
            csvData.FeatureData = new DataMatrix<int>(objectFeatures);
            return csvData;
        }
        private int[] ConvertRowData(string[] rowData)
        {
            if(rowData.Any(s => s.Length == 0))
            {
                Console.WriteLine("[WARNING] Found an empty Feature in (ConvertRowData)");
                return [];
            }

            if(SelectGrouped)
            {
                return
                [
                    (RedshiftRV[rowData[0]], AlphaRV[rowData[1]], DeltaRV[rowData[2]], ExtractNumber(rowData[8])).GetHashCode(),
                    ExtractNumber(rowData[3]),
                    ExtractNumber(rowData[4]),
                    CosmicRayActivityRV[rowData[5]],
                    ExtractNumber(rowData[6]),
                    ExtractNumber(rowData[7])
                ];
            }

            return
            [
                RedshiftRV[rowData[0]],
                AlphaRV[rowData[1]],
                DeltaRV[rowData[2]],
                ExtractNumber(rowData[3]),
                ExtractNumber(rowData[4]),
                CosmicRayActivityRV[rowData[5]],
                ExtractNumber(rowData[6]),
                ExtractNumber(rowData[7]),
                ExtractNumber(rowData[8])
            ];

            static int ExtractNumber(string number) => int.Parse(number[INT_LABEL_LENGTH..]);
        }

        public static readonly List<string> FeatureNames =
        [
            "redshift",
            "alpha",
            "delta",
            "green filter",
            "near infrared filter",
            "cosmic ray activity",
            "red filter",
            "ultraviolet filter",
            "infrared filter",
        ];

        public static readonly Dictionary<string, int> RedshiftRV = new()
        {
            { "Very Low", 0 },
            { "Low", 1 },
            { "Medium", 2 },
            { "High", 3 },
            { "Very High", 4 }
        };
        public static readonly Dictionary<string, int> AlphaRV = new()
        {
            { "Aquarius", 0 },
            { "Aries", 1 },
            { "Cancer", 2 },
            { "Capricorn", 3 },
            { "Gemini", 4 },
            { "Leo", 5 },
            { "Libra", 6 },
            { "Pisces", 7 },
            { "Sagittarius", 8 },
            { "Scorpio", 9 },
            { "Taurus", 10 },
            { "Virgo", 11 }
        };
        public static readonly Dictionary<string, int> DeltaRV = new()
        {
            { "Circumpolar North", 0 },
            { "Equatorial Region", 1 },
            { "Mid-Northern Sky", 2 },
            { "Mid-Southern Sky", 3 }
        };
        public static readonly Dictionary<string, int> CosmicRayActivityRV = new()
        {
            { "Low", 0 },
            { "Medium", 1 },
            { "High", 2 }
        };
    }
}
