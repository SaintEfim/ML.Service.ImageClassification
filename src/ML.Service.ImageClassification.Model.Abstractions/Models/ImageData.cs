using Microsoft.ML.Data;

namespace ML.Service.ImageClassification.Model.Models;

public class ImageData
{
    [LoadColumn(0)]
    public string? ImagePath;

    [LoadColumn(1)]
    public string? Label;
}
