

using System.Collections.Frozen;

namespace CS_464_HW_1
{
    public class CS_464_HW_1
    {
        public const string DatasetFolderPath = @"C:\Users\krmcd\source\repos\CS 464 HW 1\CS 464 HW 1\Dataset";

        public const string XTrainPath = DatasetFolderPath + @"\X_train.csv";
        public const string YTrainPath = DatasetFolderPath + @"\y_train.csv";
        public const string XTestPath = DatasetFolderPath + @"\X_test.csv";
        public const string YTestPath = DatasetFolderPath + @"\y_test.csv";

        public static void Main()
        {
            try
            {
                SpaceObjectDataManager objectClassifier = new(XTrainPath, YTrainPath, XTestPath, YTestPath, true);
                for (int k = objectClassifier.FeatureCount; k >= 1; k--)
                {
                    Console.WriteLine($"Selecting first k = {k} elements with the highest mutual information with the output.");
                    objectClassifier.Evaluate(true, k);
                    objectClassifier.Predict();
                    objectClassifier.ResetEvaluationData();
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
            
        }
    }
}