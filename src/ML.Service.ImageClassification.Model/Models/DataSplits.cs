using Microsoft.ML;

namespace ML.Service.ImageClassification.Model.Models;

public class DataSplits
{
    public required IDataView TrainSetRaw { get; init; }

    public required IDataView ValidationSetRaw { get; init; }

    public required IDataView TestSet { get; init; }
}
