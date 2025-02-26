using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CS_464_HW_1
{
    public class SpaceObjectDataManager
    {
        public const int Feature_COUNT = 9;
        public static readonly FeatureType[] FeatureTypes =
        [
            FeatureType.Categorical,    // redshift
            FeatureType.Categorical,    // alpha
            FeatureType.Categorical,    // delta
            FeatureType.Continious,     // green_filter
            FeatureType.Continious,     // near_infrared_filter
            FeatureType.Categorical,    // cosmic_ray_activity
            FeatureType.Continious,     // red_filter
            FeatureType.Continious,     // ultraviolet_filter
            FeatureType.Continious,     // infrared_filter
        ];

        public int EntryCount { get; private set; }
        public SpaceObjectData[] FeatureData { get; private set; }
        public int[] OutputData {  get; private set; }


        public SpaceObjectDataManager(string FeaturesPath, string outputPath)
        {
            OutputData = CSVReader.ReadRows(outputPath, false).Select(n => n.Equals("True") ? 1 : 0).ToArray();
            CSVData FeatureMatrix = CSVReader.ReadCSV(FeaturesPath, true);

            Debug.Assert(OutputData.Length == FeatureMatrix.RowCount);
            EntryCount = OutputData.Length;

            FeatureData = new SpaceObjectData[EntryCount];
            for (int row = 0; row < EntryCount; row++)
            {
                SpaceObjectData? rowData = ConvertRowData(FeatureMatrix.GetRow(row));
                if(rowData == null)
                {
                    Console.WriteLine($"Skipping row {row}.");
                    continue;
                }
                FeatureData[row] = rowData;
            }
        }
        private static SpaceObjectData? ConvertRowData(string[] rowData)
        {
            const int INT_LABEL_LENGTH = 2;

            if(rowData.Length != Feature_COUNT)
            {
                Console.WriteLine("The number of Features do not match.}");
                return null;
            }
            if(rowData.Any(s => s.Length == 0))
            {
                Console.WriteLine("Found an empty Feature.}");
                return null;
            }

            return new SpaceObjectData
            {
                redshift = RedshiftRV[rowData[0]],
                alpha = AlphaRV[rowData[1]],
                delta = DeltaRV[rowData[2]],
                green_filter = ExtractNumber(rowData[3]),
                near_infrared_filter = ExtractNumber(rowData[4]),
                cosmic_ray_activity = CosmicRayActivityRV[rowData[5]],
                red_filter = ExtractNumber(rowData[6]),
                ultraviolet_filter = ExtractNumber(rowData[7]),
                infrared_filter = ExtractNumber(rowData[8])
            };

            static int ExtractNumber(string number) => int.Parse(number[INT_LABEL_LENGTH..]);
        }

        public record SpaceObjectData
        {
            public required int redshift { get; init; }
            public required int alpha { get; init; }
            public required int delta { get; init; }
            public required int green_filter { get; init; }
            public required int near_infrared_filter { get; init; }
            public required int cosmic_ray_activity { get; init; }
            public required int red_filter { get; init; }
            public required int ultraviolet_filter { get; init; }
            public required int infrared_filter { get; init; }
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
