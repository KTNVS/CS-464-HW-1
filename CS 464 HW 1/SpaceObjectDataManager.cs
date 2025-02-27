using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        private readonly NaiveBayes NaiveBayes;

        public const int FEATURE_COUNT = 9;
        public static readonly EstimationType[] FeatureTypes =
        [
            EstimationType.Categorical,    // redshift
            EstimationType.Categorical,    // alpha
            EstimationType.Categorical,    // delta
            EstimationType.Continious,     // green_filter
            EstimationType.Continious,     // near_infrared_filter
            EstimationType.Categorical,    // cosmic_ray_activity
            EstimationType.Continious,     // red_filter
            EstimationType.Continious,     // ultraviolet_filter
            EstimationType.Continious,     // infrared_filter
        ];

        private readonly DataXY TrainData = new();
        private readonly DataXY TestData = new();

        public SpaceObjectDataManager(string trainFeaturesPath, string trainOutputPath, string testFeaturesPath, string testOutputPath)
        {
            try
            {
                TrainData = ExtractDataFromCSV(trainFeaturesPath, trainOutputPath);
                TestData = ExtractDataFromCSV(testFeaturesPath, testOutputPath);
            }
            catch { throw; }
            
            // array of object features => array of feature arrays | not done for test data as single objects outputs are predicted
            TrainData.FeatureData = TrainData.FeatureData.Transpose();
            NaiveBayes = new(TrainData.OutputData);
        }
        public void Evaluate()
        {
            Console.WriteLine("Evaluation started.");
            for (int i = 0; i < FEATURE_COUNT; i++)
                NaiveBayes.LearnFeature(new Feature(i, TrainData.FeatureData.GetRow(i), FeatureTypes[i]));
            Console.WriteLine("Evaluation completed.");
        }
        public void Predict()
        {
            Console.WriteLine("Prediction started.");
            int truePositive = 0, trueNegative = 0, falsePositive = 0, falseNegative = 0;
            for (int i = 0; i < TestData.EntryCount; i++)
            {
                int? predictedOutput = NaiveBayes.PredictFeature(TestData.FeatureData.GetRow(i));
                if(predictedOutput == null)
                {
                    Console.WriteLine($"[Warning] Output couldn't be predicted for object index {i}. Skipping.");
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
            double total = Convert.ToDouble(truePositive + trueNegative + falsePositive + falseNegative);
            Console.WriteLine($"True Positive: {(Convert.ToDouble(truePositive) / total) * 100d}%");
            Console.WriteLine($"True Negative: {(Convert.ToDouble(trueNegative) / total) * 100d}%");
            Console.WriteLine($"False Positive: {(Convert.ToDouble(falsePositive) / total) * 100d}%");
            Console.WriteLine($"False Negative: {(Convert.ToDouble(falseNegative) / total) * 100d}%");
        }

        private static DataXY ExtractDataFromCSV(string FeaturesPath, string outputPath)
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
            if (FEATURE_COUNT != FeatureMatrix.ColCount)
                throw new Exception($"[ERROR] X feature count: {FeatureMatrix.RowCount} is not equal to the number of expected features of: {outputPath.Length}");

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
        private static int[] ConvertRowData(string[] rowData)
        {
            const int INT_LABEL_LENGTH = 2;

            if(rowData.Length != FEATURE_COUNT)
            {
                Console.WriteLine("[WARNING] The number of Features do not match in (ConvertRowData)}");
                return [];
            }
            if(rowData.Any(s => s.Length == 0))
            {
                Console.WriteLine("[WARNING] Found an empty Feature in (ConvertRowData)");
                return [];
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
