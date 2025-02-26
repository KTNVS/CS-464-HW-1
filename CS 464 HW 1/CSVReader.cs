using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CS_464_HW_1
{
    public static class CSVReader
    {
        public static CSVData ReadCSV(string csvFile, bool hasLabel = true)
        {
            var data = File.ReadLines(csvFile).Skip(hasLabel ? 1 : 0).Select(line => line.Split(',')).ToArray();
            return new(data);
        }
        public static string[] ReadRows(string csvFile, bool hasLabel = true) => File.ReadLines(csvFile).Skip(hasLabel ? 1 : 0).ToArray();
    }

    public class CSVData(string[][] data)
    {
        public readonly int RowCount = data.Length;
        public readonly int ColCount = data[0].Length;

        private readonly string[][] Data = data;
        public string this[int row, int col] => Data[row][col];
        public string[] GetRow(int row) => Data[row];

        public string[] GetCol(int col)
        {
            string[] Col = new string[RowCount];
            for (int row = 0; row < RowCount; row++)
                Col[row] = Data[row][col];

            return Col;
        }
        public string[] GetCategories(int col) => GetCol(col).Distinct().Order().ToArray();
    }
}
