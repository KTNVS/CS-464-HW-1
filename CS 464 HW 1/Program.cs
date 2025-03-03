namespace CS_464_HW_1
{
    public class CS_464_HW_1
    {
        public static readonly string XTrainName = "X_train.csv";
        public static readonly string YTrainName = "y_train.csv";
        public static readonly string XTestName = "X_test.csv";
        public static readonly string YTestName = "y_test.csv";

        private static string _datasetFolderPath = "";
        private static string DatasetFolderPath
        {
            get => _datasetFolderPath;
            set
            {
                _datasetFolderPath = value;
                XTrainPath = Path.Combine(_datasetFolderPath, XTrainName);
                YTrainPath = Path.Combine(_datasetFolderPath, YTrainName);
                XTestPath = Path.Combine(_datasetFolderPath, XTestName);
                YTestPath = Path.Combine(_datasetFolderPath, YTestName);
            }
        }
        private static string XTrainPath = "", YTrainPath = "", XTestPath = "", YTestPath = "";

        public static void Main()
        {
            while (true)
            {
                Console.WriteLine("Enter the absolute path of the dataset: ");
                DatasetFolderPath = Console.ReadLine() ?? "";

                if (!Path.Exists(DatasetFolderPath))
                    Console.WriteLine("Invalid folder path.");
                else if (!Path.Exists(XTrainPath))
                    Console.WriteLine("[X_train.csv] not found.");
                else if (!Path.Exists(YTrainPath))
                    Console.WriteLine("[y_train.csv] not found.");
                else if (!Path.Exists(XTestPath))
                    Console.WriteLine("[X_test.csv] not found.");
                else if (!Path.Exists(YTestPath))
                    Console.WriteLine("[y_test.csv] not found.");
                else
                    break;
            }
            Console.WriteLine("Files found\n");
            
            SpaceObjectDataManager objectClassifier = new(XTrainPath, YTrainPath, XTestPath, YTestPath, false);
            objectClassifier.PrintMutualInformation();
            for (int k = objectClassifier.FeatureCount; k >= 1; k--)
            {
                Console.WriteLine($"Selecting first k = {k} elements with the highest mutual information with the output.");
                objectClassifier.Evaluate(true, k);
                objectClassifier.Predict();
                objectClassifier.ResetEvaluationData();
            }

        }
    }
}