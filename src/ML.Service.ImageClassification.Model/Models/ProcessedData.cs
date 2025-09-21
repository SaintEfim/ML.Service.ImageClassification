using Microsoft.ML;

namespace ML.Service.ImageClassification.Model.Models;

public class ProcessedData
{
    public required IDataView TrainSet { get; init; }

    public required IDataView ValidationSet { get; init; }
}
