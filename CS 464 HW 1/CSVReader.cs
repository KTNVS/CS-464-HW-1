namespace CS_464_HW_1
{
    public static class CSVReader
    {
        public static DataMatrix<string> ReadCSV(string csvFile, bool hasLabel = true)
        {
            try
            {
                string[][] data = File.ReadLines(csvFile).Skip(hasLabel ? 1 : 0).Select(line => line.Split(',')).ToArray();
                return new DataMatrix<string>(data);
            }
            catch (Exception ex)
            {
                throw new Exception($"[ERROR] Cannot read the CSV file with the absolute path [{csvFile}] in (ReadCSV).", ex);
            }
        }
        public static string[] ReadRows(string csvFile, bool hasLabel = true)
        {
            try
            {
                return File.ReadLines(csvFile).Skip(hasLabel ? 1 : 0).ToArray();
            }
            catch (Exception ex)
            {
                throw new Exception($"[ERROR] Cannot read the CSV file with the absolute path [{csvFile}] in (ReadRows).", ex);
            }
        }
    }
    public class DataMatrix<T>(T[][] data)
    {
        public readonly int RowCount = data.Length;
        public readonly int ColCount = data.Length == 0 ? 0 : data[0].Length;

        private readonly T[][] Data = data;

        public T this[int i, int j]
        {
            get => Data[i][j];
            set => Data[i][j] = value;
        }
        public T[] GetRow(int row) => Data[row];
        public T[] GetCol(int col) // todo: convert to enumaretor
        {
            T[] Col = new T[RowCount];
            for (int row = 0; row < RowCount; row++)
                Col[row] = Data[row][col];
            return Col;
        }
        public DataMatrix<T> Transpose()
        {
            if (RowCount == 0 || ColCount == 0)
                return DataMatrix<T>.Empty;

            T[][] transposed = new T[ColCount][];
            for (int col = 0; col < ColCount; col++)
            {
                transposed[col] = new T[RowCount];
                for (int row = 0; row < RowCount; row++)
                    transposed[col][row] = Data[row][col];
            }
            return new DataMatrix<T>(transposed);
        }
        public static readonly DataMatrix<T> Empty = new([]);
    }
}
